#!/usr/bin/env python3
"""Spike B: an L1-sharded fp32 state pool aliased by the decode op's state CB and updated in place.

On today's op (spike branch, `inplace_final_state=True`): one [K, V] shard per head core, the reader publishes the
resident CB instead of reading the state from DRAM, the compute packs the last token's state back into the shard, the
op returns the pool tensor itself. Checks: torch.equal with the fresh-tensor path; the returned tensor is the pool
(same address); a program-cache hit with the pool at a new address; 100 chained steps in place vs 100 fresh-tensor
steps; a captured trace replayed 100 times vs 100 eager in-place steps with the same inputs.
Run from the worktree root with its python_env (needs tests.ttnn...gdn_decode_test_utils).
"""
import sys
import time

import torch

import ttnn
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import SHAPES, make_inputs

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
K = V = 128
sh = SHAPES["qwen27b_tp4"]
H, HV = sh.num_key_heads, sh.num_value_heads
B = 1
BH = B * HV


def head_cores(device):
    """The factory's head -> core map: head h at (h // grid_y, h % grid_y), as a col-major CoreRangeSet."""
    gy = device.compute_with_storage_grid_size().y
    ranges = []
    full_cols, rem = divmod(BH, gy)
    for x in range(full_cols):
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, gy - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(full_cols, 0), ttnn.CoreCoord(full_cols, rem - 1)))
    return ttnn.CoreRangeSet(set(ranges))


def pool_config(device):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(head_cores(device), [K, V], ttnn.ShardOrientation.COL_MAJOR),
    )


def dev(device, t, mem=None):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


def fresh_step(device, x, s0):
    o, s = OP(
        dev(device, x["q"]),
        dev(device, x["k"]),
        dev(device, x["v"]),
        dev(device, x["g"]),
        dev(device, x["beta"]),
        initial_state=dev(device, s0),
        output_final_state=True,
    )
    return ttnn.to_torch(o), ttnn.to_torch(s)


def inplace_step(device, x, pool):
    o, s = OP(
        dev(device, x["q"]),
        dev(device, x["k"]),
        dev(device, x["v"]),
        dev(device, x["g"]),
        dev(device, x["beta"]),
        initial_state=pool,
        inplace_final_state=True,
        output_final_state=True,
    )
    return ttnn.to_torch(o), s


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=8_000_000)
    device.enable_program_cache()
    results = []
    try:
        mc = pool_config(device)
        x = make_inputs(B, 1, H, HV, K, V, seed=101)
        s0 = x["initial_state"]  # [B, HV, K, V]

        # 1. one step, in place vs fresh
        o_ref, s_ref = fresh_step(device, x, s0)
        pool = dev(device, s0.reshape(BH, K, V), mc)
        addr0 = pool.buffer_address()
        o, s = inplace_step(device, x, pool)
        same_tensor = s.buffer_address() == addr0
        eq_s = torch.equal(ttnn.to_torch(s).reshape(B, HV, K, V), s_ref)
        eq_o = torch.equal(o, o_ref)
        results.append(f"one step: state equal {eq_s}, o equal {eq_o}, returned the pool itself {same_tensor}")

        # 2. cache hit with the pool at another address
        ttnn.deallocate(pool)
        filler = dev(device, torch.zeros(BH, K, V), mc)  # occupies the old shard addresses
        pool2 = dev(device, s0.reshape(BH, K, V), mc)
        moved = pool2.buffer_address() != addr0
        t0 = time.perf_counter()
        o2, s2 = inplace_step(device, x, pool2)
        dt = time.perf_counter() - t0
        eq2 = torch.equal(ttnn.to_torch(s2).reshape(B, HV, K, V), s_ref) and torch.equal(o2, o_ref)
        results.append(
            f"cache hit, pool moved {moved} (0x{addr0:x} -> 0x{pool2.buffer_address():x}): equal {eq2}; "
            f"program cache entries {device.num_program_cache_entries()}; call {dt * 1e3:.1f} ms"
        )
        ttnn.deallocate(filler)

        # 3. 100 chained steps, in place vs fresh
        steps = 100
        state_ref = s0
        pool3 = dev(device, s0.reshape(BH, K, V), mc)
        mism = []
        xs = [make_inputs(B, 1, H, HV, K, V, seed=1000 + t, with_state=False) for t in range(steps)]
        for t in range(steps):
            o_ref, state_ref = fresh_step(device, xs[t], state_ref)
            o, st = inplace_step(device, xs[t], pool3)
            if t in (0, 9, 49, 99) and not (
                torch.equal(ttnn.to_torch(st).reshape(B, HV, K, V), state_ref) and torch.equal(o, o_ref)
            ):
                mism.append(t)
        results.append(f"100 chained steps in place vs fresh: mismatching checkpoints {mism or 'none'}")

        # 4. trace: capture one in-place step on persistent inputs, replay 100 times
        xt = make_inputs(B, 1, H, HV, K, V, seed=7, with_state=False)
        inputs = [dev(device, xt[n]) for n in ("q", "k", "v", "g", "beta")]
        pool_t = dev(device, s0.reshape(BH, K, V), mc)
        pool_e = dev(device, s0.reshape(BH, K, V), mc)

        def call(pool):
            return OP(*inputs, initial_state=pool, inplace_final_state=True, output_final_state=True)

        call(pool_t)  # compile + cache
        ttnn.to_torch(pool_t)
        pool_t2 = dev(device, s0.reshape(BH, K, V), mc)  # a fresh pool for the capture
        o_tr, s_tr = call(pool_t2)  # warm (cache hit) before capture
        # reset both pools to s0 by in-place copy? the op has no reset; start the comparison from here instead:
        base = ttnn.to_torch(pool_t2).clone()
        pool_e2 = dev(device, base, mc)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        o_tr, s_tr = call(pool_t2)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(100):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        traced = ttnn.to_torch(s_tr).clone()
        o_traced = ttnn.to_torch(o_tr).clone()
        for _ in range(100):
            o_e, s_e = call(pool_e2)
        eager = ttnn.to_torch(s_e)
        results.append(
            f"trace: 100 replays vs 100 eager in-place steps: state equal {torch.equal(traced, eager)}, "
            f"o equal {torch.equal(o_traced, ttnn.to_torch(o_e))}; replay result is the pool {s_tr.buffer_address() == pool_t2.buffer_address()}"
        )
        ttnn.release_trace(device, tid)
    finally:
        ttnn.close_device(device)
    print("\n".join(results))
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
