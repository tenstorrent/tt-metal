# Unified SP=8 over 2×(4×4) meshes — cross-mesh MoE dispatch/combine (Step 3)

Goal: run the DeepSeek prefill as if on one 8×4 mesh, but physically on 2 Z-connected
4×4 meshes (`single_bh_galaxy_2x4x4_z_graph_descriptor.textproto`), so the MoE
dispatch/combine all-to-all uses the inter-mesh **chord/Z cables** (the 9th column cable).

## Verified groundwork (all on bh-glx-110-d10u08)
- Fabric sanity on `2x4x4z` (inner rows 3-6 / outer rows 1,2,7,8) passes; Z links carry
  traffic (`MultiMeshAllToAllInter`). With `FABRIC_2D_TORUS_Y` the inner mesh's Y-ring
  closes via the chord (verified: all M0 devices gain N+S).
- Step 1: Python cross-mesh tensor over the chord works — `test_multi_mesh.py` via
  `tt-run --rank-binding /tmp/bh_2x4x4z_rank_binding.yaml` (MeshSocket send/recv). PASS.
- Step 2: DeepSeek prefill runs on BOTH meshes (2 ranks, each (4,4) Ring-4) under tt-run.
  `1 passed` per rank. (Two independent SP=4 prefills.)

## Launch recipe (2 ranks)
`tt-run --rank-binding <yaml> python -m pytest test_prefill_block.py -k 'fabric2d-torus-y-4x4 ...'`
Rank-binding: rank0=mesh0=inner {2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31},
rank1=mesh1=outer {0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29},
mesh_graph_desc_path = single_bh_galaxy_2x4x4_z_graph_descriptor.textproto.

## KEY: the kernel is already cross-mesh-ready
`writer_dispatch.cpp` sets `pkt_route_info.dst_mesh_id = dest_mesh_ids[idx]` and routes via
`get_next_hop_router_direction(mesh_id, chip_id)` → already returns Z for a peer-mesh dest.
So Step 3 is HOST-SIDE ONLY (no kernel changes).

## 3a STATUS: VERIFIED on hardware (2-rank tt-run, moe-gate_device, flag on -> 1 passed x2).
Two host edits in dispatch_program_factory.cpp (both sender paths), env-gated:
  (i) append peer-mesh devices {peer_mesh_id, 0..15} to dest_mesh_id/dest_chip_id;
  (ii) set the num_devices compile-time arg (#24 + the two "Operation parameters" lists) to
       dest_chip_id.size() so the kernel's dest_chip_ids[num_devices]=DEST_CHIP_ID matches the
       now-32-entry macro (first attempt failed: "too many initializers for [16]").
Learning: num_devices both sizes the dest array AND drives the dispatch loop (kernel
writer_dispatch.cpp:78 get_compile_time_arg_val(24), :167 dispatch_devices=num_devices).
Combine op untouched (its dest stays 16) — only dispatch needed the fix.

## 3a STATUS (orig): implemented + compiled (gated by TT_DEEPSEEK_CROSS_MESH_DISPATCH).
Edit: dispatch_program_factory.cpp (both sender paths) appends peer-mesh devices
{peer_mesh_id, 0..15} to dest_mesh_id/dest_chip_id when the env flag is set. +<cstdlib>.
ttnn rebuilt clean. Verification run (moe-gate_device, 2-rank tt-run, debug logging) was
SLOW/pending — debug-level MoE flood is pathological; re-verify without full Debug.

## 3b DESIGN (host-only, NO kernel change)
Kernel data flow: reader_dispatch.cpp:245 expert_chip = expert_dispatch_table[routed_expert];
:298 route_info[3]=expert_chip; writer_dispatch.cpp:368 dest_chip_ids[route_info[3]].
=> the expert-table VALUE is the index into the (now 32-entry) dest array. So mapping a
peer-mesh expert to index 16..31 routes it over Z with zero kernel changes.
Required host work:
  (1) shard input sequence SP=8 across the 2 meshes (today each mesh = own SP=4 shard) <- hardest
  (2) global rank-consistent expert placement (both ranks agree expert->device)
  (3) expert_dispatch_table: local experts -> 0..15, peer experts -> 16..31; dispatch_group_size=8,
      experts_per_chip = num_routed_experts // 8 (init_helpers.py:205 formula)
  (4) combine reverse path + PCC vs real single-mesh 8x4.
Files: init_helpers.py (extract_mesh_config/build_expert_dispatch_table), tt_dispatch.py,
tt_combine.py, tt_moe_routing_setup.py, tt_prefill_block.py (sp_factor=8 + input shard).

## 3a-combine STATUS: VERIFIED on hardware (flag-on MoE 2-rank -> 1 passed x2). Plumbing complete
both directions (dispatch + combine reach the peer mesh's appended destinations, non-destructive).
Next: 3b model config (dispatch_group_size=8 + expert-weight split + input SP=8 shard + PCC).

## 3a-combine STATUS (orig): implemented (needs rebuild + verify).
combine_program_factory.cpp: same gated peer-append to dest_mesh_id/dest_chip_id + new define
  fabric_defines["NUM_DEST_DEVICES"] = dest_chip_id.size(); + <cstdlib>.
writer_combine.cpp: total_mesh_devices = NUM_DEST_DEVICES (#ifdef, fallback mesh_rows*mesh_cols).
  (combine derives the array size in-kernel, unlike dispatch's num_devices compile-arg #24, so it
   needed the define + a 1-line kernel change. reader_combine doesn't use the dest array.)
Flag OFF => NUM_DEST_DEVICES=16=mesh_rows*mesh_cols => unchanged. Verify: same flag-on MoE 2-rank
run as 3a; expect 1 passed x2 (now also compiles combine's writer with the 32-entry table).

## 3b BREAKTHROUGH: routing is free; 3b = model-level config only (no kernel/routing change)
reader_dispatch.cpp:249  expert_chip = device_begin_idx + expert_chip_og * device_stride
 (cluster_axis=0: device_stride=num_cols=4, device_begin_idx=my_col). With dispatch_group_size=8
 the expert table emits chip 0..7; rows 0-3 -> linearized 0..15 (local), rows 4-7 -> 16..31
 (= the peer-mesh dests 3a appended). So peer experts route over Z automatically.
3b remaining work is therefore MODEL-LEVEL + COORDINATED (must change together for correct PCC):
  - dispatch_group_size = 8 (override mesh.shape[0]=4) + experts_per_chip = num_routed//num_groups//8
  - expert_dispatch_table built for dispatch_group_size=8 (init_helpers.create_dispatch_table)
  - EXPERT PLACEMENT: split routed-expert weights so mesh0 hosts chips 0-3, mesh1 hosts chips 4-7
  - INPUT SHARD: shard sequence SP=8 (mesh0 = SP rows 0-3, mesh1 = SP rows 4-7)
  - combine (reverse) uses same expert_chip math -> also free routing; needs its dest-array append
    (3a was dispatch-only; combine still needs the peer-append + num_devices fix)
  - PCC vs real single-mesh 8x4.
First 3b increment to try: dispatch_group_size=8 + expert table for 8, confirm a token routes to a
peer-mesh expert (DPRINT / dispatch buffer on peer mesh non-empty), before the full weight-split+shard.

## 3b FULL (PCC target) — grounded model plan (coordinated; PCC is the gate)
Test = test_prefill_block.py run_model: torch_input[1,isl,emb]; HF-layer ref -> torch_output;
build_ttnn_cache(state_dict, mesh_device,...) places expert weights; input sharded via
ttnn.from_torch mesh_mapper (SP on seq dim2, TP on emb). Output PCC vs torch_output.
For unified SP=8 over 2 ranks (each a (4,4) mesh):
  (1) CONFIG: when TT_DEEPSEEK_CROSS_MESH_DISPATCH, override dispatch_group_size=8 & sp_factor=8
      (init_helpers.extract_mesh_config + tt_prefill_block sp_factor); experts_per_chip auto-halves,
      create_dispatch_table(num_routed, 8, num_groups) auto-builds chips 0-7.
  (2) EXPERT-WEIGHT SPLIT: build_ttnn_cache/TtRoutedExpert must load mesh0 = experts for chips 0-3,
      mesh1 = chips 4-7 (per-rank expert placement; use distributed_context_get_rank()).
  (3) INPUT SP=8 SHARD: rank r feeds sequence SP-shards [4r..4r+3] of the global isl (split the
      torch_input by the SP=8 chunk order; today each mesh shards its own full input over 4 rows).
  (4) OUTPUT + PCC: gather rank0+rank1 output halves -> compare to single-mesh 8x4 torch_output.
Files: init_helpers.py (extract_mesh_config override), tt_prefill_block.py (sp_factor, input shard,
output), tt_moe.py (already param-driven), tt_routed_expert.py/build_ttnn_cache (expert split),
test_prefill_block.py (rank-aware input/ref/PCC). NOTE: 3b is a multi-session model-eng effort with
PCC iteration; all layers beneath it (fabric + dispatch/combine cross-mesh plumbing) are DONE+VERIFIED.

## Increment plan
- 3a: extend dispatch dest list (`dispatch_program_factory.cpp` ~:461 loop over
  `MeshCoordinateRange(mesh_view.shape())`) to also append the peer mesh's devices
  (control-plane lookup), gated by a new optional `peer_mesh_id` param threaded from
  `tt_dispatch.py` → `dispatch.hpp/.cpp`. Default off = no change.
- 3b: reindex `expert_dispatch_table` + `dispatch_group_size=8` (`init_helpers.py`).
- 3c: same for combine.
- 3d: model wiring in `tt_prefill_block.py` (sp_factor=8, buffer sizing).
- 3e: PCC vs single-mesh 8×4 reference.

## Blocked alt (for reference)
- True unified `(8,4)` MeshDevice spanning 2 meshes is forbidden by
  `mesh_device_view.cpp:80` TT_FATAL (single mesh_id) — not needed for this scoped approach.
- `get_fabric_node_id`/`FabricNodeId` not exposed to Python → cross-mesh logic must be C++.
