# Purpose: Run single host multi-process tests for tt_metal
mpirun-ulfm  --with-ft ulfm -np 8 $TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests --gtest_filter=FaultTolerance.ShrinkAfterRankFailure
mpirun-ulfm  --with-ft ulfm -np 8 $TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests --gtest_filter=FaultTolerance.DisableBrokenBlock
