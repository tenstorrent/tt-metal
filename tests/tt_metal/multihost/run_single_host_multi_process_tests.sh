# Purpose: Run single host multi-process tests for tt_metal
mpirun-ulfm --with-ft ulfm -np 4 $TT_METAL_HOME/build/test/tt_metal/single_host_mp_tests
