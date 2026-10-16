## Files read
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py — shapes: 4-28 local experts per chip, 70-272 tokens per expert (uniform top-k routing), ROW_MAJOR bf16 dispatch buffer, bf4 ND-sharded weights, LoFi.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp — fixed 11x8 grid; N split over 11 columns, M over 8 rows; weight sender per column at row gx % 8 multicasting down the full column; x sender per row at column (gy+1) % 11; per_core_M_max 4 (8 for kimi-k3 via the widen rule); runtime args layout for reader/writer/compute.
- kernels/adaptive_chunk.hpp — runtime chunk picker; per_core_M = ceil(count_tiles / 8) for the tail chunk, so 3-9 tile experts run per_core_M 1-2 with most rows phantom; CB blocks always move at the compile-time max (pointer-only pads), so per_core_M may vary freely.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — per K-block ready/valid handshakes for x (row) and weights (column), UP_SPLIT/DOWN_SPLIT writer co-reads, phase-4 rotating activated multicast along the row, counts broadcast at start.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp — writer reads `up` and half of `down`, and (IN1_WRITER_MCAST) does the gate/up column multicast on NoC 1; output drain row mapping.
- kernels/compute/fused_swiglu.cpp — UNPACK reads counts and mailboxes them to MATH/PACK; per-chunk m_subblocks = runtime per_core_M; matmul cost scales with per_core_M (no in1 reuse across rows).
- hybrid_routed_expert_ffn/* — has its own copies of the kernels/adaptive_chunk; only weight_runs.hpp is shared (unchanged here).
## Nodes consulted
- none (first round, empty history); baseline_1 reports (run.log + ops CSV) for per-shape counts, weight GB/s (~200-265 on every shape) and core count (88 of 110).
## Docs / external references
- none
