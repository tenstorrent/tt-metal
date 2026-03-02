#!/bin/bash
#HOSTS="UF-EV-A8-GWH01,UF-EV-A8-GWH02,UF-EV-A9-GWH01,UF-EV-A9-GWH02"
HOSTS="UF-EV-B8-GWH01,UF-EV-B8-GWH02,UF-EV-B9-GWH01,UF-EV-B9-GWH02"
ETH="ens5f0np0"

set -eo

# Reset cluster
#mpirun-ulfm --hostfile /etc/mpirun/hostfile --mca btl self,tcp --mca hwloc_base_binding_policy none --tag-output bash -c "source python_env/bin/activate && tt-smi -glx_reset --snapshot_no_tty"

# Run basic echo test
#tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml \
#    --mpi-args "--host $HOSTS --map-by rankfile:file=/etc/mpirun/rankfile --mca btl self,tcp --mca btl_tcp_if_include $ETH --bind-to none --tag-output" echo "hi"

# Run system health test
#tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml \
#    --mpi-args "--host $HOSTS --map-by rankfile:file=/etc/mpirun/rankfile --mca btl self,tcp --mca btl_tcp_if_include $ETH --bind-to none --tag-output" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="*Intermesh*"

# Run CCL tests
tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml \
  --mpi-args "--host $HOSTS \
  --map-by rankfile:file=/etc/mpirun/rankfile --mca btl self,tcp \
  --mca btl_tcp_if_include $ETH --bind-to none --tag-output" \
  bash -c "source python_env/bin/activate && pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/test_optimized_moe_decode_block.py"
#  bash -c "source python_env/bin/activate && pytest tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x16_quad_galaxy"
