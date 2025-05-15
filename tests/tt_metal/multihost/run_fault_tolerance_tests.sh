# Purpose: Run single host multi-process tests for tt_metal
${TT_METAL_HOME}/openmpi/5.0.7/bin/mpirun  --with-ft ulfm -np 8 $TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests --gtest_filter=FaultTolerance.shrink_after_rank_failure
