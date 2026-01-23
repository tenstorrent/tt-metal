#!/bin/bash
mkdir -p fabric_test_logs
HOSTS="$1"
DOCKER_IMAGE="$2"
LOG_FILE="fabric_test_logs/fabric_tests_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Running fabric tests..."
echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

./tools/scaleout/exabox/mpi-docker --image $DOCKER_IMAGE --empty-entrypoint --bind-to none --host $HOSTS -np 1 -x TT_MESH_ID=0 -x TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto -x TT_MESH_HOST_RANK=0 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml : -np 1 -x TT_MESH_ID=0 -x TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto -x TT_MESH_HOST_RANK=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml : -np 1 -x TT_MESH_ID=0 -x TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto -x TT_MESH_HOST_RANK=2 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml : -np 1 -x TT_MESH_ID=0 -x TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto -x TT_MESH_HOST_RANK=3 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml |& tee "$LOG_FILE"
