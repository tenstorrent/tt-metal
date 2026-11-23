# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop schedule and damp gates on device: the tier-1 reduced stack (tests/glm54_weights.py) as one
TtPrefillTransformer on the 8x4 mesh, running

    L0 prelude | L1 L2 (pass 1) | gate 1 | L1 L2 again, the same blocks (pass 2) | gate 2 | L3 coda

PASS-2 ATTENTION IS A STAND-IN: until two-bank sparse MLA lands (q171.2.5) a pass-2 block runs its own GLM-5.3
attention (fresh top-k, writing its KVPE slot over pass 1's for the chunk). So the reference here is the GLM-5.4
schedule with that same stand-in (``_standin_reference``), not ``glm54_stack_reference``, and the run is one
chunk. What this test owns is the schedule and the gates: which tensor each step reads, that pass 2 reruns the
pass-1 weights, and the gate math.

Checks, all on rows < ROWS (below index_topk every causal key is selected and a row depends on nothing after it,
so the CPU reference needs only those rows):

* Teacher-forced, per step: each CPU step on the device's own input against the device's output. Gates: the
  reference op at GATE_PCC on x_k and the update x_k - x_{k-1}, and the per-token beta implied by the update
  (scale check). Layers: the layer's contribution (output - input) at LAYER_DELTA_PCC, the q171.2.3 load bar.
  The block after each gate (pass-2 entry, coda entry) must also fit the gate output better than either gate
  input (attribution: catches a skipped gate, which the contribution bar alone misses after gate 2).
* Negative controls: the checks against wiring faults computed on CPU (pass 2 on another layer's weights; gate 2
  with gate 1's weights or reading x0; gate 1 reading its pass output) must fail, or they could not see them.
* Free-running: the device chain against the CPU chain, reported, no bar (synthetic weights: q171.2.3 measured
  0.953 at the pass-1 output; the bar there is on loading, not on accumulated noise).
* Determinism: a second eager forward is bit-identical, every step.
* Traced: the forward on the trace-safe metadata path captured once and replayed twice; every block output and
  every KVPE / index-cache slot is bit-identical to the same metadata forward run eagerly.
"""

import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotated_chunk_positions
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_3.block import glm_decoder_layer_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import damp_gate_reference, loop_schedule
from models.demos.deepseek_v3_d_p.reference.glm_5_4.loop import _ffn_kwargs, _mla_weights_with_indexer
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.glm54_weights import (
    GLM54ReducedConfig,
    glm54_reduced_config,
    glm54_tier1_weights,
)
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import mesh_shard_dims
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

ROWS = 512
GATE_PCC = 0.999  # approved gate bar (epic q171 DESIGN)
BETA_MAX_ABS_ERR = 0.01  # as test_glm54_damp_gate
LAYER_DELTA_PCC = 0.95  # q171.2.3's teacher-forced contribution bar (CODA_DELTA_PCC)
SP = 8
CHUNK = PREFILL_CHUNK_TOKENS_PER_CHIP * SP
TOKEN_SEED = 5
DIAG_DIR = Path.home() / ".cache" / "tt-glm54"


def _stage(msg: str, t0: float) -> None:
    logger.info(f"[stage +{time.perf_counter() - t0:7.1f}s] {msg}")


def _pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.detach().flatten().double(), b.detach().flatten().double()]))[0, 1])


def _step_names(config) -> list[tuple[str, object]]:
    """(intermediate name, LoopStep) in execution order: the device forward's names for the reference schedule."""
    names = []
    for s in loop_schedule(
        config.num_hidden_layers, config.loop_first_layer, config.loop_last_layer, config.loop_count
    ):
        if s.kind == "gate":
            names.append((f"gate{s.loop_pass}", s))
        else:
            names.append((f"layer_{s.layer}" if s.loop_pass <= 1 else f"layer_{s.layer}.pass{s.loop_pass}", s))
    return names


class _CpuSteps:
    """CPU steps on ROWS rows. Every layer call gets a fresh MLA reference (one chunk: the stand-in's pass 2 writes
    and reads only its own KVPE, as on device); shared layers take the preceding full layer's top-k."""

    def __init__(self, config, weights):
        self.config, self.layers, self.gates = config, weights.state_dict["layers"], weights.gates

    def layer(self, idx: int, x: torch.Tensor, topk=None, *, weights_of: int | None = None):
        w = self.layers[idx if weights_of is None else weights_of]
        mla = SparseMLAReference(self.config, _mla_weights_with_indexer(self.config, w["mla_weights"]), seq_len=ROWS)
        shared = self.config.indexer_types[idx] == "shared"
        out, _, tk = glm_decoder_layer_reference(
            self.config,
            None,
            w["attn_norm_weight"],
            w["ffn_norm_weight"],
            x.unsqueeze(0).to(torch.bfloat16),
            ROWS,
            indexer_topk=topk if shared else None,
            return_indexer_topk=True,
            mla_ref=mla,
            **_ffn_kwargs(w),
        )
        return out[0].float(), (topk if shared else tk)

    def gate(self, k: int, x_prev: torch.Tensor, h: torch.Tensor, *, weights_of: int | None = None):
        g = self.gates[(k if weights_of is None else weights_of) - 1]
        out, beta = damp_gate_reference(
            x_prev,
            h,
            norm_weight=g["norm_weight"],
            proj_weight=g["proj_weight"],
            proj_bias=g["proj_bias"],
            eps=self.config.rms_norm_eps,
        )
        return out.to(torch.bfloat16).float(), beta


def _standin_reference(cpu: _CpuSteps, names, x: torch.Tensor) -> dict:
    """Free-running CPU chain of the schedule with the stand-in pass-2 attention, named as the device's steps."""
    out, topk, pass_in = {}, None, None
    for name, s in names:
        if s.kind == "gate":
            x, _ = cpu.gate(s.loop_pass, pass_in, x)
            pass_in = None
        else:
            if s.loop_pass and pass_in is None:
                pass_in = x
            x, topk = cpu.layer(s.layer, x, topk)
        out[name] = x
    return out


def _implied_beta(x_prev, h, x_k):
    """Per-token scalar beta that best explains x_k - x_prev = beta * (h - x_prev) (least squares) [rows, 1]."""
    d = h - x_prev
    return ((x_k - x_prev) * d).sum(-1, keepdim=True) / (d * d).sum(-1, keepdim=True)


def _gate_check(dev_in, dev_h, dev_out, ref_out, ref_beta) -> tuple[dict, list]:
    scores = {
        "x_k": _pcc(dev_out, ref_out),
        "update": _pcc(dev_out - dev_in, ref_out - dev_in),
    }
    failures = [f"{k} PCC {v:.6f} < {GATE_PCC}" for k, v in scores.items() if v < GATE_PCC]
    beta = _implied_beta(dev_in, dev_h, dev_out)
    scores["beta"] = _pcc(beta, ref_beta)
    scores["beta_max_abs_err"] = float((beta - ref_beta).abs().max())
    if scores["beta"] < GATE_PCC:
        failures.append(f"implied beta PCC {scores['beta']:.6f} < {GATE_PCC}")
    if scores["beta_max_abs_err"] > BETA_MAX_ABS_ERR:
        failures.append(f"implied beta max abs err {scores['beta_max_abs_err']:.3e} > {BETA_MAX_ABS_ERR}")
    return scores, failures


def _layer_check(dev_in, dev_out, ref_out) -> tuple[dict, list]:
    p = _pcc(dev_out - dev_in, ref_out - dev_in)
    return {"contribution": p}, ([] if p >= LAYER_DELTA_PCC else [f"contribution PCC {p:.5f} < {LAYER_DELTA_PCC}"])


def _build(mesh_device, device_params, num_links, config, weights):
    sp_axis, tp_axis = 0, 1
    n = config.num_hidden_layers
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=GLM54ReducedConfig,
        state_dict={**weights.state_dict, "loop_gates": weights.gates},
        num_layers=n,
        seq_len=CHUNK,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        is_chunked=True,
        slot_num=1,
        max_seq_len=CHUNK,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        tp_shard_kv=True,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
    )
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=CHUNK,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=n,
        num_users=1,
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=CHUNK,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=full_indexer_rank(config, n),
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )
    return transformer, kvpe_cache, index_kv_cache


def _scalar(mesh_device, val: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _cache_tensors(kvpe_cache, index_kv_cache) -> list[ttnn.Tensor]:
    """Every device tensor the forward writes caches into (KVPE storage and the index-key cache)."""
    return [kvpe_cache.storage, index_kv_cache]


def _raw_host(t: ttnn.Tensor) -> list[torch.Tensor]:
    """Per-device host copies, bit-exact (no composition or dtype conversion beyond to_torch)."""
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM54ReducedConfig), l1_small_size=GLM53Adapter.l1_small_size
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_glm54_loop_on_device(mesh_device, device_params, num_links):
    if not is_blackhole():
        pytest.skip("GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
    t0 = time.perf_counter()
    sp = mesh_device.shape[0]
    assert sp == SP
    chunk_local = PREFILL_CHUNK_TOKENS_PER_CHIP
    config = glm54_reduced_config(CHUNK)
    assert ROWS <= config.index_topk
    weights = glm54_tier1_weights(config)
    assert all(g["proj_weight"].shape[0] == 1 for g in weights.gates), "tier 1 uses scalar gates"
    logger.info(f"tier-1 provenance: {weights.provenance}")
    _stage("weights generated", t0)
    transformer, kvpe_cache, index_kv_cache = _build(mesh_device, device_params, num_links, config, weights)
    names = _step_names(config)
    assert [(s.kind, s.loop_pass, s.layer) for s in transformer.steps] == [
        (s.kind, s.loop_pass, s.layer) for _, s in names
    ], "device schedule differs from the reference schedule"
    _stage("transformer built", t0)

    tokens = torch.randint(0, config.vocab_size, (CHUNK,), generator=torch.Generator().manual_seed(TOKEN_SEED))
    tt_tokens = ttnn.from_torch(
        torch.tensor(pack_forward_tokens([tokens.tolist()], [0], sp), dtype=torch.int64).reshape(sp, 1, chunk_local),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=mesh_shard_dims(0, 0)
        ),
    )
    positions = rotated_chunk_positions(0, sp, chunk_local)
    order = torch.tensor([positions[c][r] for c in range(sp) for r in range(chunk_local)])

    def natural(h: torch.Tensor) -> torch.Tensor:
        h = h.to(torch.float32)[0]
        nat = torch.empty_like(h)
        nat[order] = h
        return nat

    def eager():
        out = transformer.forward(
            tt_tokens,
            kvpe_cache,
            actual_isl=CHUNK,
            actual_start=0,
            actual_end=CHUNK,
            cache_user_id=0,
            return_intermediates=True,
            index_kv_cache=index_kv_cache,
        )
        ttnn.synchronize_device(mesh_device)
        return {k: natural(v) for k, v in out.items()}

    first, second = eager(), eager()
    _stage("eager forwards x2 done", t0)
    failures = []
    for name, _ in [("embed", None)] + names:
        assert torch.isfinite(first[name]).all(), f"{name} not finite"
        if not torch.equal(first[name], second[name]):
            failures.append(f"eager {name} differs between two runs")

    # --- traced: metadata path, captured once, replayed twice, against the same metadata forward run eagerly.
    metadata = ChunkMetadata(_scalar(mesh_device, 0), _scalar(mesh_device, 0), _scalar(mesh_device, CHUNK))
    taps: list[ttnn.Tensor] = []

    def meta_forward():
        taps.clear()
        transformer.forward(
            tt_tokens,
            kvpe_cache,
            actual_isl=CHUNK,
            actual_start=None,
            actual_end=None,
            cache_user_id=0,
            index_kv_cache=index_kv_cache,
            metadata=metadata,
            on_layer_hidden=lambda _layer, x: taps.append(x),
        )
        return list(taps)

    def snapshot(blocks):
        return [_raw_host(t) for t in blocks], [_raw_host(t) for t in _cache_tensors(kvpe_cache, index_kv_cache)]

    meta_eager = snapshot(meta_forward())
    ttnn.synchronize_device(mesh_device)
    _stage("metadata eager forward done", t0)
    with TracedStep(
        mesh_device,
        meta_forward,
        attach=transformer.set_trace_controller,
        detach=lambda: transformer.set_trace_controller(None),
    ) as traced:
        traced.capture()
        _stage(f"trace captured ({traced.controller.num_segments} segments)", t0)
        for rep in range(2):
            traced.replay()
            got = snapshot(traced.output)
            for kind, a, b in (("block output", got[0], meta_eager[0]), ("cache", got[1], meta_eager[1])):
                if len(a) != len(b) or not all(all(torch.equal(x, y) for x, y in zip(p, q)) for p, q in zip(a, b)):
                    failures.append(f"traced replay {rep}: {kind}s differ from the eager metadata forward")
    _stage("traced replays checked", t0)

    # --- accuracy on rows < ROWS.
    dev = {k: v[:ROWS] for k, v in first.items()}
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dev, DIAG_DIR / f"loop-device-rows{ROWS}-tokens{TOKEN_SEED}.pt")  # for diagnosis without the device
    cpu = _CpuSteps(config, weights)
    report, prev, pass_in, topk, gate_inputs = {}, "embed", None, None, None
    for name, s in names:
        if s.kind == "gate":
            ref_out, ref_beta = cpu.gate(s.loop_pass, dev[pass_in], dev[prev])
            scores, fails = _gate_check(dev[pass_in], dev[prev], dev[name], ref_out, ref_beta)
            gate_inputs = {"pass output (gate skipped)": prev, "pass input": pass_in}
            pass_in = None
        else:
            if s.loop_pass and pass_in is None:
                pass_in = prev
            ref_out, topk_out = cpu.layer(s.layer, dev[prev], topk)
            scores, fails = _layer_check(dev[prev], dev[name], ref_out)
            if gate_inputs is not None:
                # Attribution: the block after a gate must match the CPU step on the gate output better than on
                # either gate input. The contribution bar alone cannot tell gate 2's output from its pass output
                # (beta2 ~0.9 makes them close; CPU-measured 0.975 for the fault, above LAYER_DELTA_PCC).
                for label, alt in gate_inputs.items():
                    alt_pcc = _layer_check(dev[prev], dev[name], cpu.layer(s.layer, dev[alt], topk)[0])[0][
                        "contribution"
                    ]
                    scores[f"vs {label}"] = alt_pcc
                    if alt_pcc >= scores["contribution"]:
                        fails.append(f"output fits input '{alt}' ({alt_pcc:.5f}) as well as '{prev}'")
                gate_inputs = None
            topk = topk_out
        report[name] = scores
        failures += [f"{name}: {f}" for f in fails]
        prev = name
    for name, scores in report.items():
        logger.info(f"teacher-forced {name}: {scores}")
    _stage("teacher-forced checks done", t0)

    # Negative controls (CPU): each fault must fail the check above that owns it.
    controls = {
        "pass 2 on another layer's weights (L3)": _layer_check(
            dev["gate1"], dev["layer_1.pass2"], cpu.layer(1, dev["gate1"], weights_of=3)[0]
        ),
        "gate 2 with gate 1's weights": _gate_check(
            dev["gate1"],
            dev["layer_2.pass2"],
            dev["gate2"],
            *cpu.gate(2, dev["gate1"], dev["layer_2.pass2"], weights_of=1),
        ),
        "gate 2 reads x0": _gate_check(
            dev["gate1"], dev["layer_2.pass2"], dev["gate2"], *cpu.gate(2, dev["layer_0"], dev["layer_2.pass2"])
        ),
    }
    beta_h = cpu.gate(1, dev["layer_2"], dev["layer_2"])[1]
    controls["gate 1 beta reads its pass output"] = _gate_check(
        dev["layer_0"],
        dev["layer_2"],
        dev["gate1"],
        (dev["layer_0"] + beta_h * (dev["layer_2"] - dev["layer_0"])).to(torch.bfloat16).float(),
        beta_h,
    )
    for name, (scores, fails) in controls.items():
        logger.info(f"control '{name}': {scores}")
        if not fails:
            failures.append(f"control '{name}' passes: the checks cannot see it")
    _stage("negative controls done", t0)

    # Free-running, reported only.
    free_ref = _standin_reference(cpu, names, dev["embed"])
    logger.info(
        "free-running device vs stand-in reference PCC: "
        + ", ".join(f"{name} {_pcc(dev[name], free_ref[name]):.5f}" for name, _ in names)
    )
    _stage("free-running reference done", t0)
    assert not failures, failures
