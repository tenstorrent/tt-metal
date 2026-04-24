mpirun-ulfm --host $HOSTS tt-smi -glx_reset

mpirun-ulfm --host $HOSTS --mca btl_tcp_if_include ens5f0np0 --tag-output ./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path /data/scaleout_configs/SP5/cabling_descriptor.textproto --deployment-descriptor-path /data/scaleout_configs/SP5/deployment_descriptor.textproto --send-traffic --num-iterations 5

TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --tcp-interface ens5f0np0 --hosts $HOSTS --mesh-graph-descriptor /data/rsong/tt-metal/models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod_68.textproto pytest -svv models/demos/deepseek_v3_b1/tests/unit_tests/test_multi_host_pipeline.py::test_passthrough_pipeline_block
