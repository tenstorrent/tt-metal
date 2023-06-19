// combining multiple C++ source files into a single file
// to reduce the overhead of the compilation process and 
// improve build times
#include "ckernel.cc"
#include "ckernel_template.cc"
#ifdef PERF_DUMP
#include "ckernel_perf_unpack_pack.cc"
#endif
#include "ckernel_main.cc"