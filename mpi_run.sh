tt-run \
   --tcp-interface enp194s0f0np0 \
   --rank-binding $(pwd)/tests/tt_metal/distributed/config/dual_bh_lb_rank_bindings.yaml \
   --mpi-args "--host bh-lb-02,bh-lb-03" \
   bash -c "export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000 && source $(pwd)/setup.sh && pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/multihost/2xLB_4x4/test_all_links_ag.py"
