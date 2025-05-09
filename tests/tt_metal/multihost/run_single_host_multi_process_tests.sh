# Purpose: Run single host multi-process tests for tt_metal

mpirun -np 8 $TT_METAL_HOME/build/test/tt_metal/single_host_mp_tests
