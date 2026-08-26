# Migration audit — experimental/ccl + deepseek + programming_examples

Verdicts: **clean** (block maps to Pipe send/receive with no structural fight), **refactor(cost)** (in-scope but needs untangling), **defer-raw(why)** (not migrating now). Scope guard: intra-chip rectangle mcast = in scope; Ethernet/fabric/EDM/cross-chip CCL = out of scope.

## programming_examples
| kernel | verdict | notes |
|--------|---------|-------|
| `contributed/multicast/.../coordinator_kernel.cpp` | **clean** | Textbook sender: data mcast + flag mcast + barrier. Pre-handshake (counter) + data-ready (flag). Cleanest reference. Single-shot. |

## deepseek_v3_b1 (models/demos)
| kernel | verdict | notes |
|--------|---------|-------|
| `unified_kernels/mcast.hpp` | **prior-art (reference, do not migrate)** | IS the Pipe: init/op/teardown, two-sided, F3 parameterized, F2=flag, F1=flush. pre_handshake ABSENT. Strongest API evidence. |
| `unified_kernels/dataflow_utils.hpp` | **prior-art (primitive layer)** | The set_state/issue_txn helpers (unicast+multicast write, atomic_inc). Pipe should sit ON these, not replace them. |
| `unified_kernels/kv_cache_update.hpp` | **clean** | NopeSender block (150–221) maps 1:1 to Pipe send/receive built on dataflow_utils. Shared-cmd-buf branch is the only wrinkle. |
| `unified_kernels/flash_mla.hpp` | **refactor (low)** | K-chunk sender (308–367) is a clean block but lives inside a multi-branch per-chunk loop (loopback vs sharded read). Pre-handshake (counter) load-bearing. Extract the mcast block, leave loop. |
| `micro_ops/sampling/kernels/sampling_kernel.cpp` | **clean** | Loop-barrier sem-mcast (192–220). Flag-only degenerate. Needs the coord-swap helper Pipe provides. mesh_mode path is out of scope (cross-device). |

## experimental/deepseek_prefill
| kernel | verdict | notes |
|--------|---------|-------|
| `dispatch/.../reader_dispatch.cpp` | **refactor (low)** | Local idle-core mcast block (164–268) is clean F2=COUNTER. Sits in a larger reader; fabric refs minimal. Post-handshake (return-addr gather) is separate. |
| `combine/.../reader_combine.cpp` | **refactor (low)** | Same F2=counter inc_multicast block (166–246). |

## experimental/ccl — intra-chip mcast leg IN SCOPE
| kernel | verdict | notes |
|--------|---------|-------|
| `rms_allgather/.../rms_sender_reader.cpp` | **clean** | No fabric. Loopback data+flag mcast + barrier (130–155). Best loopback exemplar. |
| `rms_allgather/.../rms_writer.cpp` | **refactor (med)** | 12 fabric refs; isolate the loopback sem-mcast (97/186) from fabric leg. |
| `all_to_all_async_generic/.../all_to_all_sender_writer.cpp` | **refactor (med)** | 37 fabric refs. Loopback sem-mcast release (215–229) is in scope; rest is fabric — out of scope. |
| `moe/selective_reduce_combine/.../reader.cpp` | **refactor (low)** | Sem-mcast on NOC1 with coord-swap (108–119) in scope. |
| `moe/selective_reduce_combine/.../writer.cpp` | **refactor (med)** | 28 fabric refs; init sem-mcast (240–244) in scope, fabric leg out. |
| `llama_all_gather_matmul_async/.../reader_..._in1_ring_all_gather.cpp` | **clean** | Sem-mcast (54–58), no fabric in file. |
| `llama_all_gather_matmul_async/.../worker_receiver.cpp` | **clean** | Loopback data+flag mcast + barrier + ring pre-handshake (44–78). |
| `all_gather_concat_heads_fused/.../llama_all_gather_concat_writer.cpp` | **refactor (med)** | 25 fabric refs; two sem-mcast rectangles (197–216) in scope. |
| `moe_gpt/.../tilize_reader.cpp` | **refactor (med)** | Rich multi-rectangle data+flag mcast (660–763); value-carrying sems; chunked-burst helper. Big kernel (880 lines) but block is well-delimited. |
| `moe_gpt/.../tilize_writer.cpp` | **refactor (med)** | Local mcast helper + 3 sem-mcast sites (499/558/570). |
| `moe_compute/.../tilize_writer.cpp` | **refactor (med)** | Same helper + sem-mcast (538/716/728). |
| `moe_compute/.../tilize_reader.cpp` | **refactor (med)** | Same family as moe_gpt tilize_reader (1012 lines). |

## experimental/ccl — NOT the block / no mcast
| kernel | verdict | notes |
|--------|---------|-------|
| `all_gather_concat_heads_fused/.../llama_all_gather_concat_reader.cpp` | **n/a** | Receiver/wait side only, no mcast primitive. |
| `all_gather_concat_heads_fused/.../llama_concat_reader.cpp` | **n/a** | No mcast. |
| `all_gather_concat_heads_fused/.../tilize_writer.cpp`, `tilize_compute.cpp` | **n/a** | No mcast. |
| `rms_allgather/.../rms_receiver_reader.cpp`, `rms_compute.cpp`, `reshard_writer.hpp` | **n/a / receiver** | Receiver wait side; no mcast send. |
| `llama_all_gather_matmul_async/.../worker_writer.cpp`, `worker_reader.cpp`, `reader_..._in0_ring`, `compute/*` | **n/a** | No mcast (in0 reader has the helper but no live mcast send found). |
| `all_to_all_async_generic/.../all_to_all_sender_reader.cpp` | **defer-raw** | Mcast is fabric `to_chip_multicast` (cross-chip CCL) — **out of scope**. |

## OUT-OF-SCOPE (cross-chip CCL / fabric) — defer-raw
Every `fabric_*`, `to_chip_multicast`, `to_chip_unicast`, EDM/erisc send path across all the above CCL ops is **defer-raw: "cross-chip CCL, out of scope."** Only the intra-chip rectangle leg of each kernel was annotated.

## Counts
- **In-scope clean**: 6 — coordinator_kernel, kv_cache_update, sampling_kernel, rms_sender_reader, in1_ring reader, worker_receiver.
- **In-scope refactor**: 10 — flash_mla, reader_dispatch, reader_combine, rms_writer, all_to_all_sender_writer, selective_reduce reader, selective_reduce writer, concat_writer, moe_gpt tilize_reader/_writer, moe_compute tilize_reader/_writer (4 tilize counted as 4 → total refactor = 12; flash_mla+2 prefill+rms_writer+all_to_all+2 selred+concat = 8; +4 tilize = 12).
- **Prior-art references (not migrated)**: 2 — mcast.hpp, dataflow_utils.hpp.
- **Out-of-scope cross-chip CCL defer-raw**: all_to_all_sender_reader (fabric mcast) + the fabric leg of every refactor-tagged CCL kernel (rms_writer, all_to_all_sender_writer, selective_reduce writer, concat_writer, reader_dispatch, reader_combine).
- **n/a (no block / receiver-only)**: ~9 (concat readers, rms receiver/compute/reshard, worker_writer/reader, in0 reader, ccl compute kernels).
