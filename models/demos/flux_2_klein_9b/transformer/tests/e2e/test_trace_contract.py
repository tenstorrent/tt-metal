# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The trace contract, the host-op contract, and the depth knobs.

Three claims, each proved by running the thing rather than by reading it:

1. EVERYTHING ON DEVICE. `pipe.host_op_selftest()` runs BOTH task heads inside
   `scripts.tt_hw_planner.host_op_observer.observe_host_ops()`, with input
   construction and the one-time weight build outside the observed region and
   the whole encoded-inputs -> sample math inside it. Any host aten op fails it,
   including one hiding in the Euler update.

2. TRACE-CAPTURABLE. For every stage in `PIPELINE_STAGES`,
   `pipe.trace_capture_selftest(device)` does setup -> one untraced step ->
   `begin_trace_capture` -> the same step -> `end_trace_capture` ->
   `execute_trace` -> PCC against the untraced answer -> release. At the pinned
   capacity nothing is padded, so the traced output should be bit-identical.

3. THE DEPTH KNOBS BITE. A capped build really builds fewer blocks (and runs
   faster), while still instantiating every distinct op the full model runs --
   which is the point of the knob: profiling is per-op, so a capped build
   surfaces the same op set for a fraction of the cost.

The stage set is one phase, `denoise`, for the reason in
`e2e_plan.json::trace_contract.derivation`: the config has no `architectures`,
no `is_encoder_decoder` and no sub-configs. It is a diffusion transformer, so
there is exactly one recurring graph -- the joint text-then-image forward -- and
only the timestep scalar and the latents change between steps. Nothing here is
token-at-a-time, so no KV-cache contract applies.
"""

from __future__ import annotations

import inspect
import time

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.flux_2_klein_9b.transformer.tt import pipeline as tt_pipeline
from models.demos.flux_2_klein_9b.transformer.tt import stubs as tt_stubs

# The routing minimum: dual blocks 0/1 go to the fine-grained stubs, block 2 to
# `flux2_transformer_block` and 3+ to `encoder_stack`; single blocks 0/1 to the
# two parallel-attention stubs and 2+ to `flux2_single_transformer_block`. Below
# this a graduated aggregate would hold ZERO layers.
CAPPED_DUAL = tt_pipeline.MIN_DUAL_LAYERS
CAPPED_SINGLE = tt_pipeline.MIN_SINGLE_LAYERS


@pytest.fixture(scope="module")
def capped_pipeline(flux2_device, flux2_reference, flux2_shapes):
    """A second, deliberately shallow build on the SAME mesh, for claim 3."""
    return tt_pipeline.build_pipeline(
        flux2_device,
        model=flux2_reference,
        dual_layers=CAPPED_DUAL,
        single_layers=CAPPED_SINGLE,
        height=flux2_shapes["height"],
        width=flux2_shapes["width"],
        txt_len=flux2_shapes["txt_len"],
    )


def test_the_stage_contract_is_declared():
    """Host-only: the seams the perf engine binds exist and have the right shape."""
    assert tt_pipeline.PIPELINE_STAGES == ["denoise"], tt_pipeline.PIPELINE_STAGES

    for stage in tt_pipeline.PIPELINE_STAGES:
        for suffix in ("trace_setup", "trace_step", "trace_inputs", "trace_items"):
            name = f"{stage}_{suffix}"
            hook = getattr(tt_pipeline.Flux2KleinTransformerPipeline, name, None)
            assert callable(hook), f"missing hook {name}"

        # trace_inputs / trace_items are the ZERO-ARG seams: the engine calls
        # them with no per-model knowledge, so they must not need arguments.
        for suffix in ("trace_inputs", "trace_items"):
            params = list(
                inspect.signature(getattr(tt_pipeline.Flux2KleinTransformerPipeline, f"{stage}_{suffix}")).parameters
            )
            assert params == ["self"], f"{stage}_{suffix} must be zero-arg, got {params}"

        # trace_setup takes exactly the value trace_inputs returns.
        setup_params = [
            p
            for p in inspect.signature(
                getattr(tt_pipeline.Flux2KleinTransformerPipeline, f"{stage}_trace_setup")
            ).parameters
            if p != "self"
        ]
        assert len(setup_params) == 1, f"{stage}_trace_setup should take one argument, got {setup_params}"

    # `build_pipeline` is the module-level factory, and the standalone probe
    # hooks are module-level and zero-arg.
    assert callable(tt_pipeline.build_pipeline)
    for name in ("host_op_selftest", "trace_capture_selftest"):
        hook = getattr(tt_pipeline, name)
        assert callable(hook), name
        assert list(inspect.signature(hook).parameters) == [], f"{name} must be zero-arg"

    print(f"[flux2] stage contract: stages={tt_pipeline.PIPELINE_STAGES}, hooks and factory present", flush=True)


def test_depth_precedence_and_clamping():
    """Host-only: per-stack > stage > `layers` > full, and 0 is not a model."""
    resolve = tt_pipeline._resolve_depth

    # full depth when nothing is asked for
    assert resolve("dual_layers", 8, 4, None, None, None) == 8
    # `layers` is the default for every stack
    assert resolve("dual_layers", 8, 4, None, None, 6) == 6
    # a stage override beats `layers`
    assert resolve("dual_layers", 8, 4, None, 5, 6) == 5
    # a per-stack override beats both
    assert resolve("dual_layers", 8, 4, 7, 5, 6) == 7
    # below the routing minimum clamps UP (a graduated aggregate would be empty)
    assert resolve("single_layers", 24, 3, 1, None, None) == 3
    # zero is not a model
    assert resolve("single_layers", 24, 3, 0, None, None) == 3
    # more than the checkpoint has is capped at what it has
    assert resolve("dual_layers", 8, 4, 99, None, None) == 8
    print("[flux2] depth knobs: per-stack > stage > layers > full, clamped to the routing minimum", flush=True)


@pytest.mark.timeout(5400)
def test_a_capped_build_is_shallower_but_still_complete(capped_pipeline, flux2_pipeline):
    """Claim 3: the knob really changes what is built, and nothing distinct is lost."""
    capped, full = capped_pipeline, flux2_pipeline

    # `build_pipeline` CONSTRUCTS; it never runs the model.
    assert all(count == 0 for count in capped.invocations.values()), capped.invocations

    assert capped.dual_layers == CAPPED_DUAL and capped.single_layers == CAPPED_SINGLE, capped.depth()
    assert len(capped.dual_blocks) == CAPPED_DUAL and len(capped.single_blocks) == CAPPED_SINGLE
    assert capped.dual_layers < full.dual_layers or capped.single_layers < full.single_layers, (
        capped.depth(),
        full.depth(),
    )
    assert len(capped.single_blocks) < len(full.single_blocks), (
        len(capped.single_blocks),
        len(full.single_blocks),
    )

    # The graduated single-block stub owns single blocks 2.., so its per-step
    # call count moves with the knob -- the op/work signal really is different.
    capped_calls = capped.expected_calls_per_step()
    full_calls = full.expected_calls_per_step()
    assert capped_calls["flux2_single_transformer_block"] == CAPPED_SINGLE - 2
    assert capped_calls["flux2_single_transformer_block"] < full_calls["flux2_single_transformer_block"]

    # ... and everything OUTSIDE the two stacks is still built, so the capped
    # build runs every distinct op the full one does.
    for name in tt_stubs.GRADUATED:
        assert capped.stub_objects[name], f"{name}: absent from a capped build"
    for attr in (
        "pos_embed",
        "time_proj",
        "timestep_embedder",
        "time_guidance_embed",
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "single_stream_modulation",
        "x_embedder",
        "context_embedder",
        "norm_out",
        "proj_out",
    ):
        assert getattr(capped, attr, None) is not None, f"{attr}: not built at capped depth"

    print(
        f"[flux2] depth knob: capped dual {capped.dual_layers}/{capped.full_dual_layers}, single "
        f"{capped.single_layers}/{capped.full_single_layers} vs full {full.depth()}; all 18 stubs still present",
        flush=True,
    )


@pytest.mark.timeout(5400)
def test_the_capped_build_runs_and_is_cheaper(capped_pipeline, flux2_pipeline, flux2_inputs, flux2_device):
    """Claim 3, the runtime half: fewer blocks really is less work."""
    capped, full = capped_pipeline, flux2_pipeline

    def timed(pipe):
        pipe.reset_invocations()
        start = time.time()
        out = tt_pipeline.run_denoise_step(pipe, flux2_inputs)
        host = tt_pipeline.to_torch(out, flux2_device)
        return time.time() - start, host

    # Warm both graphs first: the very first call of any shape pays kernel
    # compilation, which would swamp the comparison.
    timed(capped)
    timed(full)
    capped_seconds, capped_out = timed(capped)
    full_seconds, full_out = timed(full)

    assert tuple(capped_out.shape) == tuple(full_out.shape), (tuple(capped_out.shape), tuple(full_out.shape))
    assert capped_seconds < full_seconds, (
        f"a {capped.dual_layers}+{capped.single_layers}-block build took {capped_seconds:.2f}s and the "
        f"{full.dual_layers}+{full.single_layers}-block build {full_seconds:.2f}s -- the depth knob is inert"
    )
    # The two builds are different models, so their outputs must differ too;
    # equal outputs would mean the capped build silently ran every layer.
    assert not torch.equal(capped_out.to(torch.float32), full_out.to(torch.float32))
    print(
        f"[flux2] depth knob: capped forward {capped_seconds:.2f}s vs full {full_seconds:.2f}s",
        flush=True,
    )


@pytest.mark.timeout(5400)
def test_the_forward_fires_no_host_op(flux2_pipeline):
    """Claim 1: zero non-benign aten ops inside BOTH task heads' forwards."""
    report = flux2_pipeline.host_op_selftest(num_steps=2)

    for head in ("denoise_step", "denoise_latents"):
        verdict = report[head]
        assert verdict["on_device"], f"{head}: {verdict['reason']}"
        assert verdict["n_host_ops"] == 0, f"{head}: {verdict['host_ops']}"
    assert report["ok"], report
    print("[flux2] host-op observer: both task heads are fully on device", flush=True)


@pytest.mark.timeout(5400)
def test_every_stage_captures_replays_and_matches(flux2_pipeline, flux2_device, flux2_shapes):
    """Claim 2: a real device trace for every stage, checked against the untraced run."""
    pipe = flux2_pipeline
    results = pipe.trace_capture_selftest(flux2_device)

    assert sorted(results) == sorted(tt_pipeline.PIPELINE_STAGES), (sorted(results), tt_pipeline.PIPELINE_STAGES)
    for stage, result in results.items():
        assert result["ok"], f"{stage}: {result.get('error') or result}"
        assert result["pcc"] is not None and result["pcc"] >= 0.99, f"{stage}: traced PCC {result['pcc']}"
        # `items` is the token count one traced step retires -- the WHOLE joint
        # sequence, because every block in the stage processes all of it.
        assert result["items"] == pipe.txt_len + pipe.img_len, (result["items"], pipe.txt_len, pipe.img_len)

    # The timestep is a persistent buffer the trace reads, rewritten ON DEVICE
    # between replays -- that is what lets one capture serve every Euler step.
    pipe.denoise_trace_set_timestep(0.25)
    written = tt_pipeline.to_torch(pipe._trace_timestep, flux2_device).to(torch.float32).reshape(-1)[0]
    assert abs(float(written) - 0.25) < 1e-6, float(written)

    print(f"[flux2] trace contract: {results}", flush=True)


@pytest.mark.timeout(5400)
def test_the_traced_shape_is_the_deployment_shape(flux2_pipeline, flux2_inputs):
    """`denoise_trace_inputs()` is zero-arg and lands exactly on the pinned capacity.

    That is the honest form of this model's padding contract: Flux2's joint
    attention takes NO attention mask (diffusers' own Flux2Pipeline passes no
    prompt mask, and the graduated attention body calls
    `scaled_dot_product_attention(is_causal=False)` with no mask argument), so a
    padded position would participate in the softmax. Masking it would mean
    editing a graduated body, which Gate 1 forbids -- so the capacity is pinned
    to the deployment length instead, and a shorter input prints the fallback.
    """
    pipe = flux2_pipeline
    trace_inputs = pipe.denoise_trace_inputs()

    assert int(trace_inputs["hidden_states"].shape[-2]) == pipe.img_len
    assert int(trace_inputs["encoder_hidden_states"].shape[-2]) == pipe.txt_len
    assert pipe.denoise_trace_items() == pipe.txt_len + pipe.img_len

    # Same shapes the e2e tests and the demos use, so the traced graph is the
    # graph the PCC was measured on.
    assert int(trace_inputs["hidden_states"].shape[-2]) == flux2_inputs["meta"]["S_img"]
    assert int(trace_inputs["encoder_hidden_states"].shape[-2]) == flux2_inputs["meta"]["S_txt"]

    ok, pcc_value = comp_pcc(flux2_inputs["hidden_states"], trace_inputs["hidden_states"], 0.999)
    assert ok, f"trace inputs are not the seeded gate inputs (correlation {pcc_value})"
    print(
        f"[flux2] trace capacity C = S_txt {pipe.txt_len} + S_img {pipe.img_len} = "
        f"{pipe.denoise_trace_items()}; nothing padded",
        flush=True,
    )
