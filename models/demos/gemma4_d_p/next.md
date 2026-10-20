**I’d skip device-driven `ttnn.slice` and build offset addressing directly.** Then separate two milestones: removing copies while preserving today’s tuning, and making lengths/configurations change within a trace.

The distinction is that **request metadata can become dynamic fairly directly; kernel configuration cannot**. Today’s Q/K chunk sizes are compile-time constants, and K splits/segmented accumulation affect core assignment, intermediate buffers, and synchronization. A descriptor containing different values won’t change the captured program. See the [compute kernel](/localdev/svuckovic/_workspace/repos/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp:37) and [configuration logic](/localdev/svuckovic/_workspace/repos/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1433).

I suggest this order:

1. **Define one request descriptor shared by all relevant operations.**

   Extend the model-owned metadata into a fixed device table with one entry per possible request:

   ```text
   active
   cache_slot
   prefix_tokens
   valid_tokens
   packed_offset_tiles     # CP-local, in 32-token rows
   packed_span_tiles       # CP-local allocated rows, including padding
   ```

   Keep **valid length separate from packed span**. A 257-token request occupies 512 packed tokens across CP8, but only 257 are valid.

   The packed offset is the same on every CP rank because each request reserves equal local capacity. The prefix determines rotated ownership. Kernels derive each rank’s valid rows from these fields.

   Initially, the host fills this table alongside the existing full-batch RoPE position tensor.

2. **Make KV updates read directly from full projected K/V buffers.**

   Give the update operation the full source buffer, descriptor table, and request index. It reads the request’s local rows and writes them to the selected cache slot using the existing stripe ownership.

   There is one additional detail: our current update requires matching source/cache dtypes. Initially, **convert the full packed K/V buffers once**, then perform offset-based updates. Fusing conversion into the update is a later optimization.

   RoPE stays exactly where it is: applied to the full batch using the staged position tensor.

3. **Add offset reads and direct output writes to SDPA together.**

   Conceptually:

   ```python
   rjsdpa(
       full_q,
       kv_cache,
       descriptors=metadata,
       request_index=i,
       output_tensor=packed_output,
       program_config=profile,
       ...
   )
   ```

   SDPA still reads K/V from the cache; the cache-update operation consumes projected K/V.

   Its Q reader uses the descriptor’s offset, and its writer places results at that same offset in the full packed output. This removes Q/K/V slice copies and output concatenation.

   **Physical buffer shape must be separate from request extent.** Passing an 8k Q buffer must not make the factory schedule an 8k request. Address calculations also need the full buffer’s head strides; this is tile-index addressing, not merely adding bytes to a DRAM pointer.

   Reuse appropriately sized statistics/scratch buffers too. Some of the currently discarded `stats` output is internal working storage. Ensure unused output rows are initialized on every replay.

   **For this milestone, keep current per-layout traces and per-request tuning.** That gives a clean measurement of the benefit from removing copies.

4. **Make request lengths and communication bounds dynamic, with tuning fixed.**

   Now change the kernels to derive active work from descriptors:

   - Query work distribution and tail masks.
   - KV write ranges.
   - Global gather and sliding halo ranges.
   - Output write ranges.

   Compile for a bounded capacity, but execute work only for the active extent. Reader, compute, writer, and communication kernels must agree on those extents.

   Start with a fixed request count per trace. Then support a fixed maximum count with inactive entries, allowing fewer requests to reuse the same trace. Inactive operations need a consistent synchronization path.

   At this point, multiple length layouts can share a trace **provided they use the same compiled configurations and fit their capacities**.

5. **Add tuning selection through a small set of profiles.**

   I would keep the configuration tuple together:

   ```text
   profile = (q_chunk, k_chunk, max_k_splits, segmented_accumulation)
   ```

   These parameters interact. In particular, segmented accumulation also affects numerical behavior, so selecting it is not purely a performance decision.

   There are three practical approaches:

   | Approach | How selection works | Trade-off |
   |---|---|---|
   | **Host selects a complete trace — recommended first** | Choose a trace with the desired per-request profiles before replay | Preserves specialized kernels; needs a bounded trace cache |
   | Several gated variants inside one trace | Capture each candidate; all its kernels check a profile ID | One trace, but extra dispatches and descriptor reads |
   | Multiple implementations inside one compiled program | Device selects a compiled code path | Largest kernel rewrite; buffer capacity and code size become concerns |

   **Precompiling variants alone is insufficient.** A normal trace replays the programs captured into it; it does not rerun Python configuration selection or search the program cache.

   With host-selected traces, selection happens **once before the full-model replay**, with no host decisions between layers. Trace keys become profile signatures and structural capacities, rather than every exact length tuple.

   I would retain specialized traces for common cases, plus a general fallback. Avoid eagerly capturing every profile combination across all request slots.

The first useful deliverable should therefore be **descriptor-based KV reads, SDPA Q reads, and packed output writes, using today’s tuning and trace layouts**. Measure that first. Then make lengths dynamic under fixed profiles, and use those measurements to decide whether device-side configuration selection is worth its complexity.

One measurement detail: the current per-request allocations happen during execution/capture setup; replay does not repeat Python allocation. The steady-state savings we’re primarily targeting are **slice/typecast/concat device work and memory traffic**.
