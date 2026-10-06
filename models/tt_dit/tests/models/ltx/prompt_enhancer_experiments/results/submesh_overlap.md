# Submesh overlap experiment — Galaxy 4x8, LTX distilled device params

Run 2026-10-06 on bh-glx-120 (tt-metal f6f9cc14e40). `full` = (4,8) submesh (what LTX uses);
`single` = (1,1) submesh at (0,0) on the same parent. Raw data: `results.json` (aliasing-first
order) and `results_guardfirst.json` (guard reservation on pristine allocators). Log: `pytest*.log`.
Addresses are per-bank DRAM offsets (8 banks); a 32 MB replicated tensor is 4 MB per bank.

| Stage | Question | Result |
|---|---|---|
| S1 | Does `create_submesh` accept a (1,1) and a (1,8) overlapping the (4,8)? | Yes. No error, 3 submeshes on the parent. Both handles report DRAM base 5,848,704. |
| S2 | Do the two handles alias DRAM on the shared chip? | Yes. `full` tensor A and `single` tensor B both at 10,043,008. After writing B, A reads 2.0 on chip 0 and 1.0 elsewhere. Each handle's buffer report lists only its own buffers. |
| S3 | Can a guard buffer on `full` keep `single` clear? | Only if it is the first allocation on `full`. Guard-first: guard at base spanning 32 MB/bank, `full`'s next tensor lands above it (43,597,440), `single`'s tensors land at 10.0M / 14.2M inside the guard, high-water 18.4M, `full` intact. Run after fragmentation: guard landed at 39.4M, both handles reused the same 10.04M hole, `full` corrupted. |
| S4 | After a trace is captured on `full`, do `single`'s allocations collide? | Yes, both directions. `single`'s first tensor landed on the trace input X (chip 0 read 7.0 instead of 1.0); replay produced Y = 15 on chip 0 vs 3 elsewhere. `single`'s second tensor landed on Y's address and was overwritten by the replay (15 instead of 9). `TT_METAL_TRACE_ALLOC_TRACKING=1` raised nothing: the tracker only sees the capturing handle's allocator. |
| S5 | Same post-capture allocation on `full` itself? | The tracker raised `RuntimeError: Found 1 device buffer(s) still alive before trace replay` before replay. The guard works within one handle, not across handles. |
| S6 | All-gather on the (1,8) row submesh | Not run (opt-in `SUBMESH_EXP_CCL=1`); moot given S2/S4. |

## Conclusion

Overlapping submeshes are legal and unprotected. Every MeshDevice handle owns an independent
allocator (`tt_metal/distributed/mesh_device.cpp:1701-1708`), so two handles over the same chip
hand out the same addresses and corrupt each other silently, including through trace replay, and
the only existing guard (the trace allocation tracker) cannot see across handles.

A device prompt enhancer therefore cannot be a sibling submesh of the DiT's (4,8) handle. Options:

1. Run it on the DiT's own (4,8) handle, the way the Gemma-3 text encoder already does: TP over
   axis 1, replicated over axis 0. One allocator, no aliasing, trace tracker covers it. Cost: the
   weights are replicated on every chip (~6 GB, affordable on 4x8) and 4 rows compute the same
   thing. Requires the gemma4 model code to accept a (4,8) mesh.
2. Reserve a region on the DiT handle with a guard buffer allocated before anything else, sized
   above the enhancer handle's lifetime high-water mark, and never freed. Works mechanically (S3
   guard-first) but is fragile: it depends on allocation order, bottom-up placement, and on no
   small top-down allocations from the sibling (one 6 KB buffer appeared near the top of DRAM
   in S3). Not recommended for production.
3. A separate box or spare chips. None on a fully used Galaxy.
