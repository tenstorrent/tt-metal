#!/usr/bin/env python3
"""Correctness battery for the multi-rectangle topk_large_indices path.

Checks, per (rows, P) config:
  - indices are a valid exact top-k set: gather(input, idx) as a multiset
    bit-matches torch.topk on the bf16-rounded input (largest, non-stable);
  - descending value order along k;
  - valid_length: no index >= valid_length even with a poisoned tail;
  - return_values arm: values output bit-matches gather(input, idx);
  - program-cache: every config runs twice (cache-hit correctness), plus a
    same-layout different-row-count pair (60 rows then 30 rows at P=4 share
    one program; distribution must re-derive via override_runtime_arguments).
Run under flock with a timeout. Single device, eager, no Tracy.
"""

import torch
import ttnn

TESTS = [
    # (rows, W, k, P or None, valid_length, return_values)
    (2, 65536, 2048, 2, None, False),
    (4, 65536, 2048, 4, None, False),
    (30, 65536, 2048, 4, None, False),  # the wave-2 target shape
    (30, 65536, 2048, 4, None, True),  # values arm
    (30, 65536, 2048, 4, 40960, False),  # valid_length bound
    (60, 65536, 2048, 4, None, False),  # same layout as 30@P4 (cache pair)
    (30, 65536, 2048, 4, None, False),  # cache-hit re-entry after 60
    (8, 65536, 512, 16, None, False),  # k=512 window, deeper trees
    (30, 65536, 2048, None, None, False),  # bare multi-row: auto rect (2x-threshold fires)
    (160, 65536, 2048, None, None, False),  # THE GLM CALL: hybrid composite (130 RP + 30 rect + concat)
    (160, 65536, 2048, None, 40960, False),  # composite with valid_length (both windows bounded)
    (160, 65536, 2048, None, None, True),  # composite values arm (two concats)
    (137, 65536, 2048, None, None, False),  # composite, odd remainder (7 rows over 30-rect capacity? r2=7)
    (2, 3000, 1536, None, None, False),  # cache-contract shape: must stay row-parallel (marginal win)
]


def bf16_round(x):
    return x.to(torch.bfloat16).to(torch.float32)


def run_case(device, rows, w, k, p, valid_length, return_values, tag):
    torch.manual_seed(hash((rows, w, k, p or 0, tag)) % (2**31))
    x = torch.randn(1, 1, rows, w, dtype=torch.float32)
    if valid_length is not None:
        # Poison the tail with huge values: any leak past valid_length is loud.
        x[..., valid_length:] = 1000.0
    xb = bf16_round(x)
    tt_x = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    kwargs = {}
    if p is not None:
        kwargs["num_slices"] = p
    if valid_length is not None:
        kwargs["valid_length"] = valid_length
    if return_values:
        out = ttnn.experimental.topk_large_indices(tt_x, k=k, return_values=True, **kwargs)
        vals_t, idx_t = out[0], out[1]
        vals = ttnn.to_torch(vals_t).to(torch.float32)
    else:
        idx_t = ttnn.experimental.topk_large_indices(tt_x, k=k, **kwargs)
        vals = None
    idx = ttnn.to_torch(idx_t).to(torch.int64)
    ttnn.deallocate(tt_x)

    ref_w = valid_length if valid_length is not None else w
    ok = True
    for r in range(rows):
        row_idx = idx[0, 0, r]
        assert row_idx.min() >= 0 and row_idx.max() < w, f"{tag} row {r}: index out of range"
        if valid_length is not None:
            assert (
                row_idx.max() < valid_length
            ), f"{tag} row {r}: index {int(row_idx.max())} >= valid_length {valid_length}"
        gathered = xb[0, 0, r, row_idx]
        ref = torch.topk(xb[0, 0, r, :ref_w], k, largest=True, sorted=True).values
        got_sorted, _ = torch.sort(gathered, descending=True)
        if not torch.equal(got_sorted, ref):
            nbad = int((got_sorted != ref).sum())
            print(f"FAIL {tag} row {r}: {nbad}/{k} multiset mismatches")
            ok = False
        #

        if not torch.all(gathered[:-1] >= gathered[1:]):
            print(f"FAIL {tag} row {r}: output not value-descending")
            ok = False
        if vals is not None and not torch.equal(bf16_round(vals[0, 0, r]), gathered):
            print(f"FAIL {tag} row {r}: values output != gather(input, idx)")
            ok = False
    return ok


def main():
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    passed = failed = 0
    try:
        for rows, w, k, p, vl, rv in TESTS:
            for trial in range(2):  # second run = program-cache hit
                tag = f"rows={rows} W={w} k={k} P={p} vl={vl} rv={rv} t{trial}"
                if run_case(device, rows, w, k, p, vl, rv, tag):
                    print(f"PASS {tag}")
                    passed += 1
                else:
                    failed += 1
    finally:
        ttnn.close_device(device)
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
