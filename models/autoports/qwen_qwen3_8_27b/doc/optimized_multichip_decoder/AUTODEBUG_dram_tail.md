# AutoFix: padding-only trailing DRAM reader

## New evidence

The parent installed the coordinate-aware native repair and reran
`installed_reader2_l0`. Its log reaches the new `optional<MeshCoordinate>`
factory and fails at `curr_storage_core_idx < num_cores_written_back` with
`Worker 7-3 has no storage area assigned`. This is a separate descriptor
construction bug; the original mesh/hop-distance assertion is resolved.

The parent reports that model-side zero-padding every rank's logical output
width to a multiple of `banks * readers * 32`, then slicing back, passes
`padded_reader2_l0.json`. This independently supports a writeback coverage/
tail issue. This repair agent did not run those device experiments.

## Verified arithmetic

For the local packed projection K5120/N4160, eight banks, two readers,
and ten storage cores:

| Quantity | Value in tiles |
| --- | --- |
| Logical N | 130 |
| Weight-bank width | 18 |
| Total DRAM width | 144 |
| Readers | 16 |
| Width per reader | ceil(130 / 16) = 9 |
| Width per output-storage core | ceil(144 / 10) = 15 |
| Used output-storage cores | ceil(130 / 15) = 9 |
| Output-storage capacity | 135 |

Workers 0 through 14 write the first 135 tiles, including five legal output
padding tiles. Worker 15 starts at column 135 and owns only DRAM padding
through column 143. The smaller-reader-than-storage branch unconditionally
requires a new storage core, even though this worker has no logical output.
For three readers, the same layout uses width 6 per reader; worker 22 is
partially clipped at capacity 135 and worker 23 has no output storage.

## Minimal native repair

The larger-reader branch already emits zero write shards when output storage
is exhausted. Extend that behavior to the smaller-reader branch:

```cpp
if (per_core_N_in1_sender < per_core_N_storage &&
    curr_storage_core_idx >= num_cores_written_back) {
    TT_FATAL(i * per_core_N_in1_sender >= N,
             "Worker {} has no output storage but still owns logical output columns", core);
    mm_in1_sender_writer_args.push_back(0);
} else if (per_core_N_in1_sender < per_core_N_storage) {
    // Existing bounded one/two-shard write assignment.
}
```

The existing common argument padding expands this zero-write case to eleven
fixed argument slots. This proves padding-only ownership before selecting
supported zero-write behavior. All workers remain in the descriptor and
continue reading and computing. The kernel waits on the output CB, loops
zero times over writes, performs its barrier, and pops the computed output
(`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:211-247`).

No kernel change, arbitrary shape allowlist, extra model tensor operation,
precision change, or global output-padding requirement is introduced.

## Isolated implementation and checks

Worktree: `/home/mvasiljevic/qwen38-full-rerun/dram-mesh-fix`.
Incremental patch: `dram_tail_worker.patch.gz`, relative to the main workspace
after applying the first coordinate repair and its regression patch.

The patch adds the explicit zero-write branch and expands the native mesh
regression with K5120/N4160/ten-storage-core cases. The test derives storage
width from the padded DRAM width, matching the observed 15-tile shard.
Descriptor inspection asserts zero write shards and all fixed argument slots
when a reader begins beyond output-storage capacity.

This agent ran a pure-Python simulation of the existing small-reader cursor
arithmetic plus the proposed exhausted-storage branch:

- Original exact case: assertion at worker 15, start 135, capacity 135.
- Fixed two-reader case: 135 tiles written, worker 15 writes zero.
- Fixed three-reader case: 135 tiles written, worker 23 writes zero.
- Exhaustive sweep: logical N 1..256, storage width 2..32, reader widths
  1..storage-1, and ceil(N/reader)+2 workers; all 126,976 cases passed
  contiguous writes, complete logical coverage, capacity bounds, and
  padding-only ownership for every zero-write worker.

This is host arithmetic evidence, not hardware/kernel execution. Black with
Python 3.12 targeting, `py_compile`, `git diff --check`, and main-workspace
`git apply --check` also passed. The native tail patch was not built by this
agent; the parent owns integration, compile/install, and hardware serialization.

After rebuilding and installing both runtime components, run:

```bash
python_env/bin/python -m pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -k test_matmul_dram_sharded_mesh_readers_cache -q
```

The regression now has twelve cases: readers 1/2/3, two output shapes, and
full/offset meshes. Then compare native unpadded reader2/reader3 layer checks
with the parent's padded control, including state/cache correctness,
changed-input trace replay, and full-layer latency.
