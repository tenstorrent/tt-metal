# Migration Audit — conv group (`mcast_pipe` / `Pipe`)

Group dirs swept:
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/`
- `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/`

All conv kernels already use the **new object API** (`Noc`, `Semaphore<>`, `MulticastEndpoint`,
`McastDst`) — no legacy free-function spellings remain. This is the substrate the `Pipe` is built
from, so migration is a *re-expression*, not an API port.

## SHARED-HEADER finding (the headline the prompt asked about)
**`conv_reader_common.hpp` does NOT host the mcast/handshake block.** It contains only activation
*read* helpers (NoC `async_read` + `async_read_barrier` + CB push/reserve): `read_sticks`,
`read_channels`, `read_activation_data`, `zero_out_tiles`, `read_sticks_activation_reuse`, etc. No
`async_write_multicast`, no `Semaphore` handshake. ⇒ **the migration unit stays per-kernel**; there
is no single shared header to swap. The closest thing to a shared block is the **file-local** pair
`multicast_data` + `mcast_block_chunked` inside `reader_conv_activations_2d_...v2.cpp` (not shared).

---

## Per-kernel verdict

| Kernel | Role | Verdict | Cost | Notes / blocker |
|---|---|---|---|---|
| `reader_writer_tiled_out_1d_mcast_sender_...` | sender | **clean** | low | Canonical `Pipe::send` (EXCLUDE_SRC/flag/no-barrier/pre_handshake). 2 call sites (weights+bias), identical. Only caveat: continuous-cell VALID assumption (see HOLE in annotation). |
| `reader_writer_tiled_out_1d_mcast_receiver_...` | receiver | **clean** | low | Canonical `Pipe::receive`. 2 call sites. |
| `writer_tiled_out_2d_mcast_sender_...` | sender | **clean (mcast part)** | low | 2D twin of 1D sender, same signature. Mcast block migrates cleanly; the `reserve_done/write_done` split-reader handshake (`L183-215`) is a SEPARATE channel — leave it. |
| `writer_tiled_out_2d_mcast_receiver_...` | receiver | **clean (mcast part)** | low | 2D twin of 1D receiver. Same `reserve_done/write_done` exclusion. |
| `activation_reader_width_sharded.cpp` | hybrid (S+R+loopback) | **refactor** | med | F3 = INCLUDE_SRC loopback; **F2-MIXED** (R→S counter `wait_min`+reset vs S→R flag `wait`+reset); F1=barrier. The `Pipe` must (a) support loopback send, (b) decide whether `wait_min`-with-reset collapses to the flag style. |
| `reader_conv_activations_2d_..._v2.cpp` | hybrid (S+R+loopback) | **refactor** | med-high | All THREE F3 sub-cases incl. the **degenerate-rect local-write (INV5)** path; **chunked send** (`mcast_block_chunked`, burst-split below `NOC_MAX_BURST_SIZE`). Both are `Pipe` features the current sketch lacks. Best single migration target to *prove the `Pipe` generality*. |
| `writer.cpp` (conv3d) | hybrid, 3 modes | **partial / defer-raw (chain mode)** | med | **mcast mode**: clean-ish `Pipe::send`/`receive` (EXCLUDE_SRC/flag/no-barrier), but receiver resets its OWN flag and uses `wait(1)`/`set(0)` count spelling — refactor to canonical. **chain mode (unicast)**: **defer-raw** — unicast forwarding chain, out of `Pipe` (mcast) scope. **drain loop** (`L140-146`): needs a `Pipe::receive` ack-only/no-consume mode. |

## Out-of-scope channels found inside block-containing kernels (do NOT swallow)
- **split-reader shared-CB handshake** (`reserve_done_sem` / `write_done_sem`): present in
  `writer_tiled_out_2d_mcast_sender`/`_receiver` and `reader_conv_activations_2d_...v2`. Local
  two-reader CB coordination, flag ping-pong, no NoC mcast. Not a `Pipe`.
- **conv3d unicast chain** + **conv3d drain loop**: see `writer.cpp` annotation.
- **conv3d `reader_vol2col.cpp` / `compute.cpp`**: declare a `semaphore_id` compile-time arg
  (`reader_vol2col.cpp:736`, `compute.cpp:218`) but **never use it** for any handshake — no block.

## Counts
- Kernels swept: conv2d 12 files, conv3d 5 files = **17**.
- Block-containing kernels: **7** (4 conv2d senders/hybrid + 2 conv2d receivers + 1 conv3d hybrid).
  - Note: 2 of the 7 (the receivers) are **grep-recall misses** — block spelled only as `Semaphore`
    methods (`.set/.up/.wait`), no `noc_*` / `async_write_multicast` token.
- Distinct mcast call sites (data+flag pairs): 1d sender 2, 2d sender 2, conv3d mcast 1,
  width-sharded 1, 2d-act-reader 1 (chunked) = **7 send sites**; receivers: 1d 2, 2d 2, conv3d 2 +
  drain 1 = **7 receive sites**.
- Verdict tally: **clean 4**, **refactor 2**, **partial/defer-raw 1**.

## Headline blockers for a single `Pipe` over the conv group
1. **F2 is NOT uniform.** Weights kernels (1d/2d) + conv3d = pure **flag** (exact `wait`+reset).
   `activation_reader_width_sharded` mixes a **`wait_min` counter** on R→S with a flag on S→R. The
   `Pipe` must pick one canonical handshake or expose both — this is exactly the F2 bake-off.
2. **F3 spans the full range** within the group: EXCLUDE_SRC (weights, conv3d), INCLUDE_SRC loopback
   (both hybrid readers), and the **degenerate-1-dest local-write dodge** (2d act reader). The `Pipe`
   MUST implement all three behind one predicate (sender ∈ rect? num_dests==0?) — the conv group is
   the strongest evidence the F3 dual/tri-path is mandatory, not optional.
3. **Chunked send below `NOC_MAX_BURST_SIZE`** (`mcast_block_chunked`) and **ack-only drain receive**
   (conv3d) are real call-site features absent from the current `Pipe` sketch.
4. **Receiver flag-reset placement differs:** conv2d receivers `set(INVALID)` BEFORE ack; conv3d
   receiver `set(0)` AFTER wait; senders re-stage (or assume continuous) VALID. The `Pipe` contract
   must nail down *who resets the level flag and when* (H3 ownership).
