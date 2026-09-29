"""Check scalar-offset chunked SDPA bounds without loading model weights.

The final K chunk covers absolute rows 4352..4607, but only rows through
4383 belong to this prefill. Its reader must load page 136 and zero-fill the
remaining seven tile rows, without consulting page-table entries 137..143.
The logical table contains only 140 entries and the cache only 4480 rows.

Unused V pages 137..139 contain +Inf in the sentinel cases. They deliberately
are not valid attention inputs: a correct reader never fetches those pages.
They test numerical isolation of unused pages, but do not by themselves prove
the reader avoided those pages: both the old and corrected readers passed this
fixture on the parent's mesh. Zero-times-Inf need not follow IEEE behavior in
the device matmul. Run on a free TP4 mesh; the caller owns serialization and
recovery.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

import ttnn

RANKS = 4
LOCAL_Q_HEADS = 8
LOCAL_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 32
START = 4096
Q_ROWS = 288
END = START + Q_ROWS
CACHE_ROWS = 4480
PAGE_COUNT = CACHE_ROWS // PAGE_SIZE
VALID_PAGES = END // PAGE_SIZE
MIN_PCC = 0.995
MAX_RELATIVE_L2 = 0.02
MAX_ABSOLUTE_ERROR = 0.02


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def compare(actual, reference):
    actual = actual.float()
    reference = reference.float()
    finite = bool(torch.isfinite(actual).all())
    stats = {
        "finite": finite,
        "nonfinite_count": int((~torch.isfinite(actual)).sum()),
        "max_abs_output": number(actual.abs().max()),
    }
    if not finite:
        return {**stats, "pass": False, "pcc": None, "relative_l2": None, "max_abs_error": None}
    left = actual.reshape(-1).double()
    right = reference.reshape(-1).double()
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    pcc = float(torch.dot(centered_left, centered_right) / (centered_left.norm() * centered_right.norm()))
    error = actual - reference
    relative_l2 = float(error.norm() / reference.norm())
    max_abs = float(error.abs().max())
    return {
        **stats,
        "pcc": number(pcc),
        "relative_l2": number(relative_l2),
        "max_abs_error": number(max_abs),
        "pass": pcc >= MIN_PCC and relative_l2 <= MAX_RELATIVE_L2 and max_abs <= MAX_ABSOLUTE_ERROR,
    }


def paged(tensor):
    return tensor.reshape(RANKS * LOCAL_KV_HEADS, PAGE_COUNT, PAGE_SIZE, HEAD_DIM).permute(1, 0, 2, 3).contiguous()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per fixture, including program-cache hits")
    parser.add_argument("--seed", type=int, default=4353)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/ifm_k2_horizon_7b/doc/optimized_full_model/chunked_attention_bound.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    torch.set_num_threads(8)
    rng = torch.Generator().manual_seed(args.seed)
    q = (torch.randn(1, RANKS * LOCAL_Q_HEADS, Q_ROWS, HEAD_DIM, generator=rng) * 0.5).bfloat16()
    k = (torch.randn(1, RANKS * LOCAL_KV_HEADS, CACHE_ROWS, HEAD_DIM, generator=rng) * 0.5).bfloat16()
    v = (torch.randn(1, RANKS * LOCAL_KV_HEADS, CACHE_ROWS, HEAD_DIM, generator=rng) * 0.5).bfloat16()
    table = torch.arange(PAGE_COUNT, dtype=torch.int32)[None]
    shuffled_table = table.clone()
    # Choose a seeded non-identity permutation; every replacement is still a
    # valid physical page ID, and only logically unused entries are modified.
    shift = int(torch.randint(1, PAGE_COUNT - VALID_PAGES, (1,), generator=rng))
    shuffled_table[:, VALID_PAGES:] = table[:, VALID_PAGES:].roll(shift, dims=-1)
    sentinel_v = v.clone()
    sentinel_v[:, :, END:, :] = float("inf")

    # Absolute-position causality, rather than top-left is_causal=True. Use
    # the exact quantized BF16 inputs, and omit inaccessible sentinel rows.
    causal = torch.arange(END)[None, :] <= torch.arange(START, END)[:, None]
    references = []
    for rank in range(RANKS):
        q_begin = rank * LOCAL_Q_HEADS
        kv_begin = rank * LOCAL_KV_HEADS
        references.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[:, q_begin : q_begin + LOCAL_Q_HEADS].float(),
                k[:, kv_begin : kv_begin + LOCAL_KV_HEADS, :END].float().repeat_interleave(4, dim=1),
                v[:, kv_begin : kv_begin + LOCAL_KV_HEADS, :END].float().repeat_interleave(4, dim=1),
                attn_mask=causal,
                dropout_p=0.0,
                is_causal=False,
            )
        )

    result = {
        "pass": False,
        "command": sys.argv,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_reader_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[4]
                / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp"
            ).read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "mesh_shape": [1, RANKS],
        "local_q_shape": [1, LOCAL_Q_HEADS, Q_ROWS, HEAD_DIM],
        "local_paged_kv_shape": [PAGE_COUNT, LOCAL_KV_HEADS, PAGE_SIZE, HEAD_DIM],
        "dtype": "bfloat16",
        "chunk_start_idx": START,
        "query_end_exclusive": END,
        "cache_rows": CACHE_ROWS,
        "page_table_logical_shape": [1, PAGE_COUNT],
        "valid_page_count": VALID_PAGES,
        "last_k_chunk_tile_ids": list(range(136, 144)),
        "expected_last_k_chunk_read_tile_ids": [136],
        "unused_v_sentinel": "+Inf only in physical pages 137..139; outside valid K/V sequence",
        "unused_table_original": table[0, VALID_PAGES:].tolist(),
        "unused_table_permuted": shuffled_table[0, VALID_PAGES:].tolist(),
        "program": {"grid": [11, 10], "q_chunk_size": 32, "k_chunk_size": 256, "exp_approx_mode": False},
        "compute": {"fidelity": "HiFi4", "math_approx_mode": False, "fp32_dest_acc_en": True, "packer_l1_acc": True},
        "thresholds": {
            "min_pcc": MIN_PCC,
            "max_relative_l2": MAX_RELATIVE_L2,
            "max_absolute_error": MAX_ABSOLUTE_ERROR,
        },
        "unused_page_invariance": "bit-exact against each rank's first finite-identity output",
        "scope": "numerical compatibility and unused-page invariance; not a direct read-bounds observation",
        "sentinel_upload_checks": [],
        "requested_repeats_per_case": args.repeats,
        "rank_outputs_checked": 0,
        "checks": [],
        "device_close_complete": False,
    }
    mesh = None
    started = time.perf_counter()
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, RANKS), trace_region_size=0)

        def to_tiles(host):
            return ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
            )

        def to_table(host):
            return ttnn.from_torch(
                host,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        tq = to_tiles(q)
        tk = to_tiles(paged(k))
        tv = to_tiles(paged(v))
        sentinel_tv = to_tiles(paged(sentinel_v))
        # Check the fixture itself: Inf must survive host conversion/upload,
        # and valid pages must remain bit-exact. This is outside timed work.
        host_sentinel = sentinel_tv.cpu(blocking=True)
        sentinel_shards = ttnn.get_device_tensors(host_sentinel)
        if len(sentinel_shards) != RANKS:
            raise AssertionError(f"Expected {RANKS} sentinel shards, got {len(sentinel_shards)}")
        paged_v = paged(v)
        for rank, shard in enumerate(sentinel_shards):
            uploaded = ttnn.to_torch(shard)
            begin = rank * LOCAL_KV_HEADS
            valid_exact = torch.equal(uploaded[:VALID_PAGES], paged_v[:VALID_PAGES, begin : begin + LOCAL_KV_HEADS])
            unused_inf = bool(torch.isposinf(uploaded[VALID_PAGES:]).all())
            result["sentinel_upload_checks"].append(
                {"rank": rank, "valid_pages_exact": valid_exact, "unused_pages_positive_inf": unused_inf}
            )
            if not valid_exact or not unused_inf:
                raise AssertionError(f"Sentinel fixture did not survive upload on rank {rank}")
        tt = to_table(table)
        shuffled_tt = to_table(shuffled_table)
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10), q_chunk_size=32, k_chunk_size=256, exp_approx_mode=False
        )
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        baselines = []
        for case, value_tensor, table_tensor in (
            ("finite_identity", tv, tt),
            ("unused_inf_identity", sentinel_tv, tt),
            ("unused_inf_permuted_table", sentinel_tv, shuffled_tt),
        ):
            for repeat in range(args.repeats):
                output = ttnn.transformer.chunked_scaled_dot_product_attention(
                    tq,
                    tk,
                    value_tensor,
                    table_tensor,
                    chunk_start_idx=START,
                    program_config=program,
                    compute_kernel_config=compute,
                )
                # Fence/read the parent mesh before extracting host shards.
                host_output = output.cpu(blocking=True)
                shards = ttnn.get_device_tensors(host_output)
                if len(shards) != RANKS:
                    raise AssertionError(f"Expected {RANKS} host shards, got {len(shards)}")
                for rank, shard in enumerate(shards):
                    actual = ttnn.to_torch(shard)
                    reference = references[rank]
                    if actual.shape != reference.shape:
                        raise AssertionError(f"Rank {rank}: expected shape {reference.shape}, got {actual.shape}")
                    if case == "finite_identity" and repeat == 0:
                        baselines.append(actual.clone())
                    invariant = torch.equal(actual, baselines[rank])
                    stats = compare(actual, reference)
                    row = {
                        "case": case,
                        "repeat": repeat,
                        "rank": rank,
                        **stats,
                        "exact_unused_page_invariance": invariant,
                        "pass": stats["pass"] and invariant,
                    }
                    result["checks"].append(row)
                    result["rank_outputs_checked"] += 1
                    print(json.dumps(row, allow_nan=False), flush=True)
                output.deallocate(True)
        result["pass"] = all(row["pass"] for row in result["checks"])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        # Persist findings even if a device-close error follows the reads.
        write_result(args.output, result)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_close_complete"] = True
            write_result(args.output, result)

    assert result["pass"], f"Chunked attention bound regression failed; see {args.output}"


if __name__ == "__main__":
    main()
