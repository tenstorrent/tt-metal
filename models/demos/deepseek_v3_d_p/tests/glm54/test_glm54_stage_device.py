# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 four-stage placement on device (q171.2.8): the tier-1 reduced stack (tests/glm54_weights.py) as one
TtPrefillRuntime per stage on one 8x4 mesh, chained in process the way the runner chains ranks over D2D:

    G0 L0 | G1 L1 L2 pass 1 + gate 1 | G2 L1 L2 pass 2 + gate 2 (own weights, bank-A copy) | G3 L3

plus the undivided runtime (all four layers, both passes) the stages are compared with. The reduced schedule is
the 106-pass one in small: 6 positions, split (0,1) (1,2) (3,2) (5,1) by ``pass_schedule``.

PASS-2 ATTENTION IS THE GLM-5.3 STAND-IN until q171.2.5: G2's 'shared' loop layers attend at the injected pass-1
top-k and write their own pass-2 KV over the chunk's bank-A rows; the undivided runtime's pass 2 recomputes its
top-k. So past pass 1 the stages are not expected to equal the undivided run; what is checked there is the wiring.

Checks, two chunks (a full one, then a partial one reading the first from cache), eager:

* Pass 1 is the undivided model's: G0's and G1's block outputs bit-identical to the undivided runtime's.
* Each stage runs its own gate: G1's output fits damp gate 1 on (G1 input, G1's last block output) at the gate bar
  and not gate 2's weights; G2's output fits gate 2 and not gate 1's weights.
* Handoff: the KVPE rows G1 sends, written at the chunk's metadata into a fresh copy, equal G1's whole cache after
  every chunk; on chunk 0 G2's own cache just before its forward equals G1's (bank A, written by the runtime); G2
  receives G1's top-k set.
* Layer acks: G0 0, G1 1 2, G2 5 6 (layer + 4: pass 2's own ids), G3 3; the undivided runtime 0 1 2 5 6 3.
* Finite outputs; a second eager chain is bit-identical.

Traced: the four stages rebuilt with use_trace, captured, replayed for the same two chunks (records as the runner
forwards them); G0/G1 outputs bit-identical to the eager chain's, G2/G3 finite and repeat-identical, G1's traced
send of the KVPE rows bit-exact against its cache.
"""

import gc
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import damp_gate_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import pass_schedule, stage_indexer_types
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.glm54_weights import (
    GLM54ReducedConfig,
    glm54_reduced_config,
    glm54_tier1_weights,
)
from models.demos.deepseek_v3_d_p.tt.loop_handoff import write_kvpe_copy
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

SP = 8
CHUNK = PREFILL_CHUNK_TOKENS_PER_CHIP * SP
CHUNKS = [(0, CHUNK), (CHUNK, CHUNK + 3008)]  # (start, end): full, then partial reading the first from cache
SEQ = 2 * CHUNK
GATE_PCC = 0.999
FORMAT = MlaKvCacheFormat.SCALED_FP8
TOKEN_SEED = 11
STAGES = {"G0": (0, 1), "G1": (1, 2), "G2": (3, 2), "G3": (5, 1)}  # pass positions
ACKS = {"U": [0, 1, 2, 5, 6, 3], "G0": [0], "G1": [1, 2], "G2": [5, 6], "G3": [3]}


def _stage(msg: str, t0: float) -> None:
    logger.info(f"[{time.perf_counter() - t0:7.1f}s] {msg}")


def _pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def _raw(t: ttnn.Tensor) -> list[torch.Tensor]:
    """Per-device host copies, bit-exact."""
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _host(mesh_device, t: ttnn.Tensor) -> torch.Tensor:
    """[rows, hidden] fp32: rows concatenated over SP, hidden over TP (row order is the chips' order; the gate is
    row-wise, so it needs none)."""
    h = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    )
    return h.reshape(-1, h.shape[-1]).float()


class _AckSink:
    def __init__(self):
        self.layers = []

    def layers_completed(self, start, end, *_):
        self.layers += list(range(start, end))


class _Rig:
    """One runtime per stage (and the undivided one) on the mesh, with its caches, taps and ack sink."""

    def __init__(self, mesh_device, device_params, num_links, config, weights, *, use_trace: bool, undivided: bool):
        self.mesh, self.config, self.use_trace = mesh_device, config, use_trace
        schedule = pass_schedule(config)
        names = (["U"] if undivided else []) + list(STAGES)
        self.rt, self.caches, self.sinks, self.taps = {}, {}, {}, {}
        for name in names:
            first_pos, n_pos = (0, len(schedule)) if name == "U" else STAGES[name]
            rs = schedule.rank_slice(first_pos, n_pos)
            loop_passes = None if name == "U" else rs.loop_passes
            rt_config = TtPrefillRuntimeConfig(
                num_layers=rs.num_layers,
                max_seq_len=SEQ,
                mesh_shape=tuple(mesh_device.shape),
                sp_axis=0,
                tp_axis=1,
                chunk_size=CHUNK,
                num_users=1,
                num_links=num_links,
                topology=per_axis_topology(device_params["fabric_config"]),
                capacity_factor=8,
                gate_fallback_mode=GateComputeMode.DEVICE_FP32,
                model_cfg=GLM54ReducedConfig,
                routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
                sparse_kv_cache_format=FORMAT,
                use_trace=use_trace,
                tp_shard_kv=True,
                first_layer_idx=rs.first_layer,
                is_first_rank=first_pos == 0,
                is_last_rank=first_pos + n_pos == len(schedule),
                loop_passes=loop_passes,
                loop_gates=weights.gates,
            )
            sd = weights.state_dict
            state_dict = {
                "embed_weight": sd["embed_weight"],
                "layers": sd["layers"][rs.first_layer : rs.first_layer + rs.num_layers],
            }
            rt = TtPrefillRuntime(mesh_device, config, state_dict, rt_config)
            types = stage_indexer_types(config, loop_passes or ())
            n_index = sum(1 for t in types[rs.first_layer : rs.first_layer + rs.num_layers] if t == "full")
            slots = rt_config.cache_num_users
            self.caches[name] = MlaKvCaches(
                kvpe=init_mla_kv_cache(
                    cache_format=FORMAT,
                    hf_config=config,
                    mesh_device=mesh_device,
                    seq_len=SEQ,
                    mesh_shape=list(mesh_device.shape),
                    sp_axis=0,
                    tp_axis=1,
                    num_kvpe_cache_layers=rs.num_layers,
                    num_users=slots,
                    bank_aligned=True,
                ),
                index=None
                if n_index == 0
                else init_kvpe_cache(
                    kvpe_cache_head_dim=config.index_head_dim,
                    mesh_device=mesh_device,
                    seq_len=SEQ,
                    mesh_shape=list(mesh_device.shape),
                    sp_axis=0,
                    tp_axis=1,
                    num_kvpe_cache_layers=n_index,
                    num_users=slots,
                    dtype=ttnn.bfloat8_b,
                    bank_aligned=True,
                ),
            )
            rt.compile(self.caches[name])
            self.sinks[name] = _AckSink()
            rt.set_layer_completion_sink(self.sinks[name])
            if use_trace:
                rt.capture_trace(self.caches[name])
            else:
                taps = self.taps[name] = []
                rt._on_layer_hidden = lambda layer, x, taps=taps: taps.append((layer, _raw(x)))
            self.rt[name] = rt

    def release(self):
        for rt in self.rt.values():
            rt.release_trace()
            rt.model.release_sub_device_managers()
        self.rt.clear()
        self.caches.clear()
        gc.collect()


def _record(rt, slot, start, end):
    return rt._meta_words_dev((slot, start, end))


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
def test_glm54_stages_on_device(mesh_device, device_params, num_links):
    if not is_blackhole():
        pytest.skip("GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
    t0 = time.perf_counter()
    mesh_device.enable_program_cache()
    config = glm54_reduced_config(SEQ)
    weights = glm54_tier1_weights(config)
    tokens = torch.randint(0, config.vocab_size, (SEQ,), generator=torch.Generator().manual_seed(TOKEN_SEED)).tolist()
    _stage("weights generated", t0)
    failures = []

    # ---------------- eager ----------------
    rig = _Rig(mesh_device, device_params, num_links, config, weights, use_trace=False, undivided=True)
    rt = rig.rt
    _stage("eager runtimes built and compiled", t0)
    g2_forward = rt["G2"].model.forward
    g2_bank_a = {}

    def g2_forward_snapshot(*args, **kwargs):
        g2_bank_a.setdefault("before", _raw(rig.caches["G2"].kvpe.storage))
        return g2_forward(*args, **kwargs)

    rt["G2"].model.forward = g2_forward_snapshot
    sent_sets, got_sets = [], []
    g2_pc = rt["G2"].prefill_chunk

    def g2_prefill_chunk(*args, loop_handoff=None, **kwargs):
        got_sets.append([_raw(t) for t in loop_handoff[1:]])
        return g2_pc(*args, loop_handoff=loop_handoff, **kwargs)

    rt["G2"].prefill_chunk = g2_prefill_chunk
    shadow = init_mla_kv_cache(
        cache_format=FORMAT,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=SEQ,
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        tp_axis=1,
        num_kvpe_cache_layers=2,
        num_users=rig.rt["G2"].config.cache_num_users,
        bank_aligned=True,
    )

    def eager_chain(tag: str) -> dict:
        outs = {}
        for name in rig.sinks:
            rig.sinks[name].layers.clear()
            rig.taps[name].clear()
        for c, (start, end) in enumerate(CHUNKS):
            chunk_tokens = tokens[start:end] + [0] * (CHUNK - (end - start))
            kw = dict(slot_id=0, actual_start=start, actual_end=end)
            rt["U"].prefill_chunk(rt["U"].make_chunk_input(chunk_tokens), rig.caches["U"], **kw)
            x0 = rt["G0"].prefill_chunk(rt["G0"].make_chunk_input(chunk_tokens), rig.caches["G0"], **kw)
            outs[f"G0.{c}"] = _raw(x0)
            x0_host = _host(mesh_device, x0)
            g1 = rt["G1"].prefill_chunk(x0, rig.caches["G1"], **kw)
            x1, kvpe_rows, *sets = g1
            outs[f"G1.{c}"] = _raw(x1)
            outs[f"G1.cache.{c}"] = _raw(rig.caches["G1"].kvpe.storage)
            sent_sets.append([_raw(t) for t in sets])
            x1_host = _host(mesh_device, x1)
            md = tuple(rt["G1"]._meta1_dev(v) for v in (0, start, end))
            write_kvpe_copy(shadow.storage, kvpe_rows, md, rt["G1"]._loop_handoff, first_kv_layer=0, num_kv_layers=2)
            outs[f"shadow.{c}"] = _raw(shadow.storage)
            x2 = rt["G2"].prefill_chunk(x1, rig.caches["G2"], loop_handoff=[kvpe_rows, *sets], **kw)
            outs[f"G2.{c}"] = _raw(x2)
            x2_host = _host(mesh_device, x2)
            rt["G3"].prefill_chunk(x2, rig.caches["G3"], **kw)
            ttnn.synchronize_device(mesh_device)
            outs[f"gate1.{c}"] = (x0_host, rig_last_tap_host(rig, "G1"), x1_host)
            outs[f"gate2.{c}"] = (x1_host, rig_last_tap_host(rig, "G2"), x2_host)
            outs[f"taps.{c}"] = {name: list(rig.taps[name]) for name in rig.taps}
            for name in rig.taps:
                rig.taps[name].clear()
        outs["acks"] = {name: list(s.layers) for name, s in rig.sinks.items()}
        _stage(f"eager chain {tag} done", t0)
        return outs

    def rig_last_tap_host(rig, name):
        # The stage's last block output (its pass output h_k), from the per-device raw copies.
        layer, per_dev = rig.taps[name][-1]
        rows, cols = mesh_device.shape
        grid = [torch.cat(per_dev[r * cols : (r + 1) * cols], dim=-1) for r in range(rows)]
        h = torch.cat(grid, dim=-2)
        return h.reshape(-1, h.shape[-1]).float()

    first = eager_chain("1")
    rt["G2"].model.forward = g2_forward  # one bank-A snapshot (chunk 0) is enough
    second = eager_chain("2")

    # Acks: every pass once per chunk, pass 2 under its own ids.
    for name, ids in ACKS.items():
        if first["acks"][name] != ids * len(CHUNKS):
            failures.append(f"{name} acked {first['acks'][name]}, expected {ids} per chunk")
    for c in range(len(CHUNKS)):
        taps = first[f"taps.{c}"]
        # Pass 1 equals the undivided model, bit for bit: G0's L0, G1's L1 L2.
        u = taps["U"]
        if not (_same(taps["G0"][0][1], u[0][1]) and taps["G0"][0][0] == u[0][0] == 0):
            failures.append(f"chunk {c}: G0 L0 differs from the undivided L0")
        for i in range(2):
            if not (_same(taps["G1"][i][1], u[1 + i][1]) and taps["G1"][i][0] == u[1 + i][0]):
                failures.append(f"chunk {c}: G1 block {i} differs from the undivided pass-1 block")
        # Each stage's own gate.
        for k, own, other in ((1, 0, 1), (2, 1, 0)):
            x_prev, h, x_k = first[f"gate{k}.{c}"]
            if not all(torch.isfinite(t).all() for t in (x_prev, h, x_k)):
                failures.append(f"chunk {c}: gate {k} inputs/outputs not finite")
                continue
            for w, must_pass in ((weights.gates[own], True), (weights.gates[other], False)):
                ref, _ = damp_gate_reference(x_prev, h, **{k: v.float() for k, v in w.items()}, eps=config.rms_norm_eps)
                scores = (_pcc(x_k, ref), _pcc(x_k - x_prev, ref - x_prev))
                ok = min(scores) >= GATE_PCC
                label = f"gate-{1 + (w is weights.gates[1])} weights"
                logger.info(f"chunk {c} stage gate {k} vs reference with {label}: x_k / update PCC {scores}")
                if ok != must_pass:
                    failures.append(f"chunk {c}: stage gate {k} {'misses' if must_pass else 'fits'} {label}")
        # Handoff.
        if not _same(first[f"shadow.{c}"], first[f"G1.cache.{c}"]):
            failures.append(f"chunk {c}: KVPE rows G1 sends do not rebuild its cache")
        if not all(_same(a, b) for a, b in zip(got_sets[c], sent_sets[c])):
            failures.append(f"chunk {c}: G2 received other top-k sets than G1 sent")
    if not _same(g2_bank_a["before"], first["G1.cache.0"]):
        failures.append("chunk 0: G2's bank A before its forward differs from G1's cache")
    # Determinism.
    for key, val in first.items():
        if key.startswith(("G", "shadow")) and not _same(val, second[key]):
            failures.append(f"eager {key} differs between two chains")
        if key.startswith("G2.") and not all(torch.isfinite(t.float()).all() for t in val):
            failures.append(f"{key} not finite")
    _stage(f"eager checks done ({len(failures)} failures so far)", t0)
    rig.release()
    del rig, rt, shadow
    gc.collect()

    # ---------------- traced ----------------
    rig = _Rig(mesh_device, device_params, num_links, config, weights, use_trace=True, undivided=False)
    rt = rig.rt
    _stage("traced runtimes built, compiled, captured", t0)
    traced = {}
    for sink in rig.sinks.values():
        sink.layers.clear()  # acks of the warm-up and capture passes
    for rep in range(2):
        for c, (start, end) in enumerate(CHUNKS):
            chunk_tokens = tokens[start:end] + [0] * (CHUNK - (end - start))
            x0 = rt["G0"].prefill_chunk(
                rt["G0"].make_chunk_input(chunk_tokens),
                rig.caches["G0"],
                None,
                None,
                None,
                metadata_msg=_record(rt["G0"], 0, start, end),
            )
            traced[f"G0.{c}.{rep}"] = _raw(x0)
            g1 = rt["G1"].prefill_chunk(
                ttnn.clone(x0), rig.caches["G1"], None, None, None, metadata_msg=_record(rt["G1"], 0, start, end)
            )
            x1, kvpe_rows, *sets = g1
            traced[f"G1.{c}.{rep}"] = _raw(x1)
            shadow = init_mla_kv_cache(
                cache_format=FORMAT,
                hf_config=config,
                mesh_device=mesh_device,
                seq_len=SEQ,
                mesh_shape=list(mesh_device.shape),
                sp_axis=0,
                tp_axis=1,
                num_kvpe_cache_layers=2,
                num_users=rt["G2"].config.cache_num_users,
                bank_aligned=True,
            )
            md = tuple(rt["G1"]._meta1_dev(v) for v in (0, start, end))
            write_kvpe_copy(shadow.storage, kvpe_rows, md, rt["G1"]._loop_handoff, first_kv_layer=0, num_kv_layers=2)
            if c == 0 and not _same(_raw(shadow.storage), _raw(rig.caches["G1"].kvpe.storage)):
                failures.append(f"traced rep {rep}: G1's traced KVPE send does not rebuild its cache")
            x2 = rt["G2"].prefill_chunk(
                ttnn.clone(x1),
                rig.caches["G2"],
                None,
                None,
                None,
                metadata_msg=_record(rt["G2"], 0, start, end),
                loop_handoff=[ttnn.clone(t) for t in [kvpe_rows, *sets]],
            )
            traced[f"G2.{c}.{rep}"] = _raw(x2)
            rt["G3"].prefill_chunk(
                ttnn.clone(x2), rig.caches["G3"], None, None, None, metadata_msg=_record(rt["G3"], 0, start, end)
            )
            ttnn.synchronize_device(mesh_device)
    for c in range(len(CHUNKS)):
        for name in ("G0", "G1"):
            if not _same(traced[f"{name}.{c}.0"], first[f"{name}.{c}"]):
                failures.append(f"traced {name} chunk {c} differs from the eager chain")
        for name in ("G0", "G1", "G2"):
            if not _same(traced[f"{name}.{c}.0"], traced[f"{name}.{c}.1"]):
                failures.append(f"traced {name} chunk {c} differs between two replays")
        if not all(torch.isfinite(t.float()).all() for t in traced[f"G2.{c}.0"]):
            failures.append(f"traced G2 chunk {c} not finite")
        if not _same(traced[f"G2.{c}.0"], first[f"G2.{c}"]):
            logger.info(f"traced G2 chunk {c} differs from eager (stand-in pass 2; reported only)")
    for name, ids in ACKS.items():
        if name != "U" and rig.sinks[name].layers != ids * 2 * len(CHUNKS):
            failures.append(f"traced {name} acked {rig.sinks[name].layers}, expected {ids} per chunk")
    _stage(f"traced checks done ({len(failures)} failures)", t0)
    rig.release()
    assert not failures, failures
