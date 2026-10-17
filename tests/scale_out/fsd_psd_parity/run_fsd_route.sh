#!/usr/bin/env bash
# Experiment C: FSD-route rank-binding solve (the local FM stand-in).
#
# Runs generate_rank_bindings so that the PhysicalSystemDescriptor is built OFFLINE from a
# Factory System Descriptor (FM's own ported builder, build_physical_descriptor_from_file,
# issue #52859) instead of from live hardware discovery. One -np 1 process builds the whole-
# system PSD centrally, exactly like FM does, then feeds the SAME topology mapper tt-run uses.
#
# Requires the experimental #52859 wiring applied + rebuilt:
#   cd /workspace/group/tt-metal
#   git apply /workspace/group/fsd_psd_parity/experiment_C/experiment_C_52859.patch   # (tree is not git; use patch -p1 < ... if needed)
#   ninja -C build_Release generate_rank_bindings
#
# Args: MGD_NAME (e.g. 4x4_bh_torus_xy_mesh_graph_descriptor)  OUT_DIR
set -u
export TT_METAL_HOME=/workspace/group/tt-metal
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib:${LD_LIBRARY_PATH:-}"
MGD_NAME="${1:-4x4_bh_torus_xy_mesh_graph_descriptor}"
OUT="${2:-/workspace/group/fsd_psd_parity/experiment_C/fsd_route/out}"
FSD="$TT_METAL_HOME/tests/scale_out/4x_bh_quietbox/factory_system_descriptors/factory_system_descriptor_4x_bh_quietbox.textproto"
MGD="$TT_METAL_HOME/tests/scale_out/4x_bh_quietbox/mesh_graph_descriptors/${MGD_NAME}.textproto"
EXE="$TT_METAL_HOME/build_Release/tools/scaleout/generate_rank_bindings"
rm -rf "$OUT"; mkdir -p "$OUT"
timeout 120 mpirun --allow-run-as-root --oversubscribe -np 1 \
  -x "TT_METAL_HOME=$TT_METAL_HOME" -x "TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME" \
  -x "TT_METAL_USE_MGD_2_0=1" \
  -x "TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH=$FSD" \
  "$EXE" -m "$MGD" --output-dir "$OUT" > "$OUT/run.log" 2>&1
echo "rc=$? -> $OUT/rank_bindings.yaml"
