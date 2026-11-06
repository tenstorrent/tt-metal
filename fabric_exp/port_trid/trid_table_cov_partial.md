| case | rule (routers) | uniform (routers) |
| --- | --- | --- |
| bh_1d | 32: G4 8gr 8@0 (x80) | 32: G4 8gr 16@0 (x80) |
| bh_1d_1erisc | 32: G4 8gr 8@0 (x40) | 32: G4 8gr 16@0 (x40) |
| bh_1d_mux | 32: G4 8gr 8@0 (x40) | 32: G4 8gr 16@0 (x40) |
| bh_1d_trim | 32: G4 8gr 8@0 (x80) | 32: G4 8gr 16@0 (x80) |
| bh_2d | 16: G4 4gr 4@0 (x80) | 16: G4 4gr 16@0 (x80) |
| bh_2d_1erisc | 16: G4 4gr 4@0 (x40) | 16: G4 4gr 16@0 (x40) |
| bh_2d_mux | 16: G4 4gr 4@0 (x40) | 16: G4 4gr 16@0 (x40) |
| bh_2d_trim | 16: G4 4gr 4@0 (x80) | 16: G4 4gr 16@0 (x80) |
| bh_2d_udm_gtest | - | 16: G4 4gr 16@0 (x40) |
| bh_2d_vc2 | 16: G4 4gr 4@0 | 4: G1 4gr 4@4 (x80) | 16: G4 4gr 4@0 | 4: G1 4gr 4@4 (x80) |
| bh_2d_vc2_1erisc | 16: G4 4gr 4@0 | 4: G1 4gr 4@4 (x40) | 16: G4 4gr 4@0 | 4: G1 4gr 4@4 (x40) |
| bh_neighbor | 32: G4 8gr 8@0 (x80) | 32: G4 8gr 16@0 (x80) |
| bh_neighbor_1erisc | 32: G4 8gr 8@0 (x40) | 32: G4 8gr 16@0 (x40) |
| bh_ring | 32a: G1 32gr 16@0! (x80) | 32a: G1 32gr 8@0! (x80) |
| bh_ring_1erisc | 32a: G1 32gr 16@0! (x40) | 32a: G1 32gr 8@0! (x40) |
| bh_ring_mux | 32a: G1 32gr 16@0! (x40) | 32a: G1 32gr 8@0! (x40) |
| bh_torus | 16a: G1 16gr 16@0 (x96) | 16a: G1 16gr 8@0! (x96) |
| bh_torus_1erisc | 16a: G1 16gr 16@0 (x48) | 16a: G1 16gr 8@0! (x48) |
| bh_torus_vc2 | 16: G4 4gr 4@0 | 4: G1 4gr 4@4 (x48); 16a: G1 16gr 8@0! | 4: G1 4gr 4@8 (x32) | 16: G4 4gr 4@0 | 4: G1 4gr 4@4 (x48); 16a: G1 16gr 8@0! | 4: G1 4gr 8@8 (x32) |
| bh_trim_gtest | - | 16: G4 4gr 16@0 (x752); 16a: G1 16gr 8@0! (x384); 32: G4 8gr 16@0 (x320) |
| bh_vc1 | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x96) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x96) |
| bh_vc1_1erisc | 8: G2 4gr 4@0 (x2); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x44); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 4: G1 4gr 4@4 (x2) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x48) |
| bh_vc1_vc2 | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 2@8 (x96) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 4@8 (x96) |
| bh_vc1_vc2_1erisc_O3 | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x1); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 2@8 (x46); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 2@8 | 2: G1 2gr 2@8 (x1) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 4@8 (x48) |
| bh_vc1_vc2_O3 | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 2@8 (x96) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 4@8 (x96) |
| ci_2x8 | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x448) |
| ci_2x8_pt | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x448) |
| ci_het_2x4 | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x64); 8a: G1 8gr 8@0 | 4: G1 4gr 8@8 (x80) |
| ci_het_bigmesh | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x191); 8a: G1 8gr 8@0 | 4: G1 4gr 8@8 (x143); 8a: G1 8gr 8@0 | 4: G1 4gr 8@8 | 4: G1 4gr 4@4 (x1) |
| ci_het_combo | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x144); 8a: G1 8gr 8@0 | 4: G1 4gr 8@8 (x112) |
| ci_nsew | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x208) |
| ci_trim_z | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x184) |
| glx2_mesh | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x848) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x848) |
| glx2_mesh_novc | 16: G4 4gr 4@0 (x848) | 16: G4 4gr 16@0 (x848) |
| glx2_mesh_vc2_O3 | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 4@8 (x848) |
| glx3_mesh | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x448) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x448) |
| glx3_mesh_vc2_O3 | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 4@8 (x448) |
| glx3_passthrough | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x400) | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x448) |
| glx3_torus_yx | - | 8: G2 4gr 4@0 (x1); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x111); 8a: G1 8gr 8@0 | 4: G1 4gr 8@8 (x95); 8a: G1 8gr 8@0 | 4: G1 4gr 8@8 | 4: G1 4gr 4@4 (x1) |
| glx_1d | 32: G4 8gr 8@0 (x416) | 32: G4 8gr 16@0 (x416) |
| glx_1x32_line | 32: G4 8gr 8@0 (x248) | 32: G4 8gr 16@0 (x248) |
| glx_1x32_ring | 32a: G1 32gr 16@0! (x256) | 32a: G1 32gr 8@0! (x256) |
| glx_2d | 16: G4 4gr 4@0 (x416) | 16: G4 4gr 16@0 (x416) |
| glx_express | 16a: G1 16gr 16@0 (x544) | 16a: G1 16gr 8@0! (x544) |
| glx_express_1erisc | 16a: G1 16gr 16@0 (x272) | 16a: G1 16gr 8@0! (x272) |
| glx_express_ew | 16a: G1 16gr 16@0 (x544) | 16a: G1 16gr 8@0! (x544) |
| glx_ring | 32a: G1 32gr 16@0! (x512) | 32a: G1 32gr 8@0! (x512) |
| glx_torus | 16a: G1 16gr 16@0 (x512) | 16a: G1 16gr 8@0! (x512) |
| glxz_mesh | - | 8: G2 4gr 4@0 (x1); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x182); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 4: G1 4gr 4@4 (x1) |
| glxz_mesh_novc | - | 16: G4 4gr 16@0 (x184) |
| glxz_mesh_vc2_O3 | - | 8: G2 4gr 4@0 | 4: G1 4gr 4@4 (x1); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 4@8 (x182); 8: G2 4gr 4@0 | 4: G1 4gr 4@4 | 2: G1 2gr 4@8 | 2: G1 2gr 4@8 (x1) |
