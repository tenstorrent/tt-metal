# Halo Buffer Design — fabric-only NeighborPad + CONV3D_H_HALO

## Goal

Eliminate the dominant bottleneck in the WAN 2.2 VAE decoder on BH Loud Box (2×4 mesh, 8×P150b):
before this change NeighborPad took 203ms (58% of wall time) for 480p. The root cause was a
~117MB DRAM interior copy that NeighborPad did on every call to assemble the full padded tensor
before handing it to conv3d.

## Decisions

| Decision | Reason | Rejected Alternative |
|---|---|---|
| fabric_only NP: skip interior copy, write only halo rows | Interior copy (117MB DRAM→DRAM per NP call) was the dominant cost, not the fabric CCL transfer (~17ms). Eliminating it removes ~186ms per NP call. | Keep interior copy, optimize with async prefetch — still pays the DRAM bandwidth |
| Compact halo buffer layout `[H-top \| H-bot \| W-left \| W-right]` | Single contiguous allocation, simple base+offset indexing in the reader kernel | Separate per-direction buffers — more allocations and harder to pass addresses to kernel |
| Conv3d reads interior from original unpadded tensor | Original tensor already lives in DRAM; no copy needed. With fabric_only NP the padded tensor no longer exists. | Allocate a new padded tensor with only halos filled, interior zeroed — adds allocation + write overhead |
| 2-CQ sub-device isolation (SD0=4 fabric, SD1=116 compute) | TT-Metal enforces exclusive CQ ownership per sub-device. Two ops on different CQs concurrently requires non-overlapping sub-devices. | Single CQ sequential — simpler but forfeits any NP/conv3d overlap |
| Load/unload halo sub-device manager per WanCausalConv3d.forward call | Regular ops (RMSNorm, add, etc.) use all 120 cores; they cannot run while the 2-sub-device halo manager is active. clear_loaded_sub_device_manager() is a fast host-side state change. | Persistent halo manager — impossible because sub-devices must be non-overlapping and full-grid sub-device cannot coexist with fabric+compute sub-devices |
| Inter-CQ event ordering (CQ1 records NP-done, CQ0 waits) | Conv3d must not start reading halo data until NP has finished writing it. Events are the correct dispatch-time ordering primitive. | In-kernel semaphore spin — writer increments only its own L1; reader cores on different Tensix cores poll their own stale copy, causing deadlock |
| compute_output_dims uses effective_padding = internal_padding + h_halo_padding | internal_padding is (0,0,0) for spatially-parallel conv3d. Without the halo adjustment, H_out = H_in−2 (valid conv) instead of H_in (same-pad conv). Both compute_output_specs (output tensor shape) and the program factory (kernel loop bounds) need the same fix. | Pass external padding directly to conv3d — would require changing the public op interface and hashing |

## Constraints & Workarounds

- **Hardware target:** BH Loud Box, 2×4 mesh (8×P150b, Blackhole architecture)
- **Sub-device non-overlap:** TT-Metal validates at create_sub_device_manager time that no two sub-devices share a core. A "full grid" sub-device cannot coexist with fabric + compute sub-devices in the same manager.
- **No diagonal halo exchange:** The compact W-halo buffer covers rows `[0, H_dev-1]` only; it does not include the H-halo rows (`h=-1` and `h=H_dev`). This means corner cells `(h_outside AND w_outside)` are approximated (see Surprises section).
- **OOM at 720p without halo buffer:** The full padded interior tensor at 720p (~224MB per device per tensor) does not fit. The halo buffer (only boundary rows) is required for 720p to run at all, not just for speed.
- **T_out_block > 1 required to activate halo path:** `use_h_halo_buffer` is only set when `_needs_halo AND T_out_block > 1`. Layers with T_out_block=1 still use the full NP path. On 480p 2×4, the k333 layers at each upsampling stage have T_out_block=3 and use the halo path; k311 temporal convs and k133 spatial reshapes do not.

## Surprises & Discoveries

- **Corner approximation mismatch:** In the old full-NP path, W-NP ran after H-NP and overwrote corner cells `(h=-1, w=-1)` etc. with W-neighbor's H-padded edge column. In the new path, `gather_rows_halo` gives h_outside priority over w_outside, so corners read from the H halo buffer with w clamped. These produce different (both approximate) values because neither path has the diagonal neighbor's data. The difference is negligible (99.976% end-to-end PCC) but could be resolved by extending the W-halo buffer to cover `H_dev + 2*padding_h` rows and adjusting corner priority.
- **load_sub_device_manager is non-trivial:** It enqueues reset commands to all CQs (dispatch wait + go_signal_mcast for CQ0). These commands are asynchronous from the host but serialize against in-flight kernels on the device. This means the sub-device switch implicitly drains prior work, providing correct ordering with no extra synchronize calls.
- **build_metal.sh install step required:** cmake --build only writes to build_Release/ttnn/. Python imports from ttnn/ttnn/. Running cmake --build without the install step leaves stale binaries, causing confusing "already fixed" bugs.

## Open Questions

- [ ] Can T-slice pipelining (true NP/conv3d overlap within a single call) be achieved? Requires in-kernel global semaphore polling across all 116 reader cores — needs NOC-based polling rather than local L1 spin. Currently the CQ1→CQ0 event makes them sequential.
- [ ] Should the W-halo buffer be extended to cover H-halo rows so corner approximation matches the old path exactly?

## State

- [x] fabric_only NeighborPad: skips interior copy, writes compact halo buffer
- [x] CONV3D_H_HALO reader: gather_rows_halo reads H/W boundaries from compact buffer
- [x] compute_output_dims fix: uses effective_padding including h_halo_padding
- [x] 2-CQ sub-device manager: SD0=fabric, SD1=compute, non-overlapping
- [x] Inter-CQ event ordering: NP done on CQ1 → conv3d starts on CQ0
- [x] Load/unload halo manager per WanCausalConv3d.forward
- [x] PCC verified: 99.976% end-to-end 480p 2×4 BF16 (threshold 99.9%)
- [ ] T-slice pipelining (NP/conv3d true concurrency within a call)
- [ ] Corner approximation fix (extend W-halo to include H-halo rows)

## Key Measurements

| Config | Wall time | ms/frame | Notes |
|---|---|---|---|
| 480p 2×4 main branch (T_out_block=1, full NP) | 1.49s | 18.4ms | Baseline |
| 480p 2×4 our branch (T_out_block>1, full NP, no halo) | 1.36s | 16.7ms | T_out_block sweep only |
| 480p 2×4 our branch (T_out_block>1 + halo buffer) | 1.20s | 14.8ms | Full improvement |
| 720p 2×4 main branch | OOM | — | Full padded tensor too large |
| 720p 2×4 our branch | 2.58s | 31.9ms | Halo buffer required for 720p |

T=21 latent frames (81 video frames), uncached, BF16, fake weights, BH Loud Box 2×4.

Reproduce:
```bash
# 480p
python run_vae_decoder_ablation.py  # H=60 W=104 in script
# 720p
# set H=90 W=160 TARGET_HEIGHT=720 TARGET_WIDTH=1280 in run_vae_decoder_ablation.py
python run_vae_decoder_ablation.py
```
