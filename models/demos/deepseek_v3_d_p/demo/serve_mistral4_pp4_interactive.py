# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Interactive chat server for Mistral Small 4, built as the REAL PP=4 x (8,1) stage geometry.

WHAT THIS IS, AND WHAT GAP IT CLOSES
-------------------------------------
Same re-prefill-per-token trick as `serve_mistral4_interactive.py` (see that module's docstring
for the full explanation of why deepseek_v3_d_p has no decode path and generates by re-running
the whole prefill per token). The only difference is how the model is built: four submeshes of
8 chips each (SP=8 x TP=1, 9 layers per stage), carved from the (8,4) galaxy exactly as
`test_prefill_pipeline_concurrent.py`'s throughput benchmark carves them, instead of one 36-layer
model on the whole mesh.

Existing PP=4 coverage has a real gap this closes:
  - `test_prefill_pipeline_stages.py` checks layer-slicing correctness, but runs all four
    "stages" on the FULL (8,4) mesh at TP=4 (deliberately, so the pretrained weights are
    byte-identical to single-rank and the token comparison is exact) -- it never touches the
    real (8,1) TP=1 submesh geometry.
  - `test_prefill_pipeline_concurrent.py` measures the real (8,1) geometry's throughput, but
    every stage is headless (`is_last_rank=False`) and the input is a synthetic random token
    tensor -- it never runs a real tokenizer, a real chat template, or samples more than a
    probability-and-token-id pair to compare against single-rank.
Neither exercises "a real chat session, through the real API, on the real stage geometry,
producing human-legible multi-token output." This does.

EAGER ON PURPOSE -- THIS IS A FUNCTIONAL CHECK, NOT A THROUGHPUT DEMO
----------------------------------------------------------------------
Unlike the single-rank demo, this does NOT trace-capture. The point here is "does the actual
production stage geometry answer sensibly through the real serving path," at the small windows
used for interactive testing -- not raw speed, which `test_prefill_pipeline_concurrent.py`
already measures properly (traced, concurrent, many requests in flight). Every generated token
pays four sequential EAGER forwards plus three host hand-off round-trips -- the same "host"
transport `test_prefill_pipeline_concurrent.py`'s `PP_HANDOFF=host` path measures the cost of,
just at a much smaller window here.

Measured on this 12 kW machine, window 1024: token 1 costs ~110s (all four stages JIT-compile
their programs; this is a one-time per-process cost, same as the single-rank demo's own ~60-100s
startup, just paid on the first generated token here instead of at server startup), then steady
state is **~0.47s/token (~2.1 tok/s)**. Do not average a short reply -- the first-token compile
cost dominates anything under ~20-30 tokens and makes the demo look far slower than it is; judge
on a long enough reply for the average to converge, exactly as this project's docs repeatedly
warn about elsewhere. ~2.1 tok/s is roughly 4x slower than the traced single-rank demo's ~9
tok/s, which is a smaller gap than pure eager-vs-traced would predict (traced is ~20x faster than
eager elsewhere in this project) -- removing the TP-axis collectives is presumably buying back
some of what eager dispatch and the host hand-offs cost. Tracing this the way the single-rank
demo does (per-submesh segmented capture, one-iteration lag between stages) would likely close
most or all of the remaining gap; that is real, separate follow-up work, not a "minimal" one.

RUN
---
    cd /data/kmabee/tt-metal
    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
    export LD_LIBRARY_PATH=$PWD/build_Release/lib:$LD_LIBRARY_PATH
    export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
    export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_pp
    export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=/data/kmabee/mistral4_caches/ref_cache
    export PREFILL_SERVE_SEQ_LEN=1024 PORT=8000
    ./python_env/bin/pytest models/demos/deepseek_v3_d_p/demo/serve_mistral4_pp4_interactive.py \
        -k "serve" -s

Then talk to it exactly as the single-rank demo (same API, same client):
    curl -N http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' -H 'Authorization: Bearer dummy' \
      -d '{"model":"mistral-small-4","stream":true,"max_tokens":16,
           "messages":[{"role":"user","content":"Name three French cities."}]}'

Note the cache: `ttnn_cache_pp`, not `ttnn_cache_8x4` -- the PP stage weight layout is different
from the single-rank one (see MISTRAL_SMALL4_BRINGUP.md's PP section).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.demo.serve_mistral4_interactive import PORT, _build_app
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache

SP, TP, PP = 8, 1, 4
TOTAL_LAYERS = 36
LAYERS_PER_STAGE = TOTAL_LAYERS // PP
SERVE_SEQ_LEN = int(os.environ.get("PREFILL_SERVE_SEQ_LEN", 1024))


class PP4TokenGenerator:
    """One-token-at-a-time generator over four real (8,1) PP stages, eager (untraced).

    Duck-types `PrefillTokenGenerator`'s interface (`.tokenizer`, `.isl_total`, `.generate()`)
    so it plugs into `serve_mistral4_interactive._build_app` unchanged -- only how a token gets
    produced differs. Each step runs the four stages in sequence exactly as
    `test_prefill_pipeline_stages.py` validates (same weights, same layer slicing), except the
    stages live on four SEPARATE submeshes rather than sharing one mesh, so each stage's output
    activation has to be re-sharded onto the next stage's submesh through the host in between.
    """

    def __init__(self, *, stages, kvs, subs, tokenizer, isl_total: int, padding_side: str):
        assert padding_side == "right", (
            f"serve requires right padding (LM head reads row actual_isl-1); got '{padding_side}'. "
            "Run the serve entry with the right_pad tokenizer."
        )
        self.stages = stages
        self.kvs = kvs
        self.subs = subs
        self.tokenizer = tokenizer
        self.isl_total = isl_total
        self._temperature = 0.0

        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 11
        self.stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
        logger.info(f"serve(pp4): stop token ids = {sorted(self.stop_ids)}, pad id = {self.pad_id}")

    def _upload(self, host_ids: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            host_ids.reshape(SP, 1, self.isl_total // SP),
            device=self.subs[0],
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.subs[0], dims=(0, None), mesh_shape=(SP, TP)),
        )

    def _handoff(self, activation: ttnn.Tensor, src_sub, dst_sub) -> ttnn.Tensor:
        """Re-shard one stage's output onto the next stage's submesh, through the host.

        This is the same HANDOFF="host" transport test_prefill_pipeline_concurrent.py measures
        the cost of (there: 42 MB/hop at its 5,120 window, ~1121 ms/iteration). Correct, and the
        known-slow part of staying minimal here -- a real transport is separate follow-up work.
        """
        host = ttnn.to_torch(
            activation, mesh_composer=ttnn.ConcatMesh2dToTensor(src_sub, dims=(2, 3), mesh_shape=(SP, TP))
        )
        return ttnn.from_torch(
            host.to(torch.bfloat16),
            device=dst_sub,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(dst_sub, dims=(2, 3), mesh_shape=(SP, TP)),
        )

    def _forward_token(self, window: torch.Tensor, n: int) -> int:
        handoff = self._upload(window)
        token_id = None
        for r, (stage, kv, sub) in enumerate(zip(self.stages, self.kvs, self.subs)):
            out = stage.forward(handoff, kv, actual_isl=n, temperature=self._temperature)
            if r < PP - 1:
                next_handoff = self._handoff(out, sub, self.subs[r + 1])
                ttnn.deallocate(out)
                handoff = next_handoff
            else:
                token_id, _prob, _ = out
        return int(token_id)

    def generate(self, prompt_ids: list[int], max_tokens: int, temperature: float):
        """Yield (token_id, text_piece, seconds_for_this_token) until stop/limit/window-full."""
        n = len(prompt_ids)
        if n >= self.isl_total:
            raise ValueError(
                f"prompt is {n} tokens but the served window is {self.isl_total}. "
                f"Raise PREFILL_SERVE_SEQ_LEN (multiple of {64 * SP}) or shorten the prompt."
            )

        window = torch.full((1, self.isl_total), self.pad_id, dtype=torch.int64)
        window[0, :n] = torch.tensor(prompt_ids, dtype=torch.int64)

        for _ in range(max_tokens):
            if n >= self.isl_total:
                logger.warning(f"serve(pp4): hit the {self.isl_total}-token window, stopping generation")
                break
            t0 = time.time()
            self._temperature = temperature
            token_id = self._forward_token(window, n)
            dt = time.time() - t0
            logger.info(f"serve(pp4): token {n - len(prompt_ids) + 1} in {dt:.2f}s (actual_isl={n})")

            if int(token_id) in self.stop_ids:
                logger.info(f"serve(pp4): stop token {int(token_id)} after {n - len(prompt_ids)} generated tokens")
                break

            window[0, n] = int(token_id)
            n += 1
            piece = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            yield int(token_id), piece, dt


@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 bring-up targets Blackhole")
@pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE, l1_small_size=768),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"])
@pytest.mark.timeout(0)  # a server runs until killed
def test_serve_pp4(variant, config_only, mesh_device, device_params, weight_cache_path, tokenizer, request):
    """Serve Mistral Small 4 for interactive chat, built as the real PP=4 x (8,1) stage geometry."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained TTNN cache unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    for mod in ("fastapi", "uvicorn"):
        pytest.importorskip(mod, reason=f"{mod} not in python_env: ./python_env/bin/pip install fastapi uvicorn")
    import uvicorn

    min_multiple = 64 * SP
    if SERVE_SEQ_LEN % min_multiple != 0:
        pytest.fail(
            f"PREFILL_SERVE_SEQ_LEN={SERVE_SEQ_LEN} must be a multiple of {min_multiple} "
            f"(64 tokens/chip for the MoE masked_bincount grid x sp={SP}); 512 is the minimum"
        )

    config = config_only
    config.max_seq_len = SERVE_SEQ_LEN
    subs = mesh_device.create_submeshes(ttnn.MeshShape(SP, TP))
    assert len(subs) == PP, f"expected {PP} ({SP},{TP}) submeshes from {mesh_device.shape}, got {len(subs)}"
    cache_path = weight_cache_path / f"{SP}x{TP}"

    stages, kvs = [], []
    try:
        for r, sm in enumerate(subs):
            stages.append(
                TtPrefillTransformer(
                    mesh_device=sm,
                    config=config,
                    model_cfg=MistralSmall4Config,
                    state_dict={},  # weights come from the TTNN cache
                    num_layers=LAYERS_PER_STAGE,
                    seq_len=SERVE_SEQ_LEN,
                    dispatch_buffer_capacity_factor=8,
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    sp_axis=0,
                    tp_axis=1,
                    is_balanced=False,
                    gate_fallback_mode=GateComputeMode.GPT_DEVICE,
                    weight_cache_path=cache_path,
                    lm_head_is_column_parallel=True,
                    routing_use_l1_small_for_semaphores=True,
                    first_layer_idx=r * LAYERS_PER_STAGE,
                    is_first_rank=(r == 0),
                    is_last_rank=(r == PP - 1),  # unlike the throughput benchmark: this one needs a real token
                )
            )
            kvs.append(
                init_mla_kv_cache(
                    cache_format=MlaKvCacheFormat.BFP8_TILE,
                    hf_config=config,
                    mesh_device=sm,
                    seq_len=SERVE_SEQ_LEN,
                    mesh_shape=(SP, TP),
                    sp_axis=0,
                    num_kvpe_cache_layers=LAYERS_PER_STAGE,
                )
            )
        logger.info(f"serve(pp4): built {PP} stages of {LAYERS_PER_STAGE} layers each, window={SERVE_SEQ_LEN}")

        gen = PP4TokenGenerator(
            stages=stages,
            kvs=kvs,
            subs=subs,
            tokenizer=tokenizer,
            isl_total=SERVE_SEQ_LEN,
            padding_side=tokenizer.padding_side,
        )
        app = _build_app(gen)
        logger.success(
            f"Mistral Small 4 PP=4 x (8,1) prefill-only chat server on :{PORT} | window={SERVE_SEQ_LEN} tokens "
            f"| {PP} stages x {LAYERS_PER_STAGE} layers | eager (untraced) -- expect seconds/token, not fast"
        )
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
    finally:
        for stage in stages:
            stage.release_sub_device_managers()
