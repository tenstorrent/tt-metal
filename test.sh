#!/bin/bash
#SBATCH --nodes=4
#SBATCH --nodelist=wh-glx-a04u02,wh-glx-a05u02,wh-glx-a05u08,wh-glx-a05u14
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err

source python_env/bin/activate
export TT_METAL_HOME="/data/aliu/tt-metal"
export PYTHONPATH="/data/aliu/tt-metal"
export TT_METAL_LOGGER_TYPES=Fabric
export TT_METAL_LOGGER_LEVEL=debug
mpirun-ulfm --host wh-glx-a04u02,wh-glx-a05u02,wh-glx-a05u08,wh-glx-a05u14 --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0 --tag-output echo "hi"
#mpirun-ulfm --host wh-glx-a04u02,wh-glx-a05u02,wh-glx-a05u08,wh-glx-a05u14 --mca bl self,tcp --mca btl_tcp_if_include ens5f0np0 --tag-output ./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path /data/scaleout-configs/4xWH_4x32/cabling_descriptor.textproto --deployment-descriptor-path /data/scaleout-configs/4xWH_4x32/deployment_descriptor.textproto
tt-run --rank-binding tests/tt_metal/distributed/config/4x32_quad_galaxy_rank_bindings.yaml --mpi-args "--host wh-glx-a04u02,wh-glx-a05u02,wh-glx-a05u08,wh-glx-a05u14 --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0 --bind-to none --tag-output" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.Test4x32QuadGalaxyFabric1DSanity" |& tee out.txt
