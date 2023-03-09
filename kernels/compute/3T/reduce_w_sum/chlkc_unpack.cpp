#include <cstdint>
#include "llk_unpack_common.h"
#include "llk_unpack_reduce.h"
namespace NAMESPACE
{

void unpack_main()
{
uint32_t int_scaler = get_compile_time_arg_val(0);

union {
    uint32_t i;
    float f;
} u = {int_scaler}; // Need to do this trick to interpret scaler as float
uint32_t Ht = get_compile_time_arg_val(1);
uint32_t Wt = get_compile_time_arg_val(2);
uint32_t NC = get_compile_time_arg_val(3);
float scaler = u.f;

llk_setup_operands();
llk_unpack_reduce_init<PoolType::SUM,ReduceDim::REDUCE_ROW>();
llk_unpack_reduce_hw_configure_disaggregated<PoolType::SUM,ReduceDim::REDUCE_ROW>(0,scaler);
for (uint32_t nc = 0U; nc < NC; nc++) {
  constexpr int onetile = 1;
  int reduce_dst_idx = 0;
  for (uint32_t ht = 0U; ht < Ht; ++ht) {
    for (uint32_t wt = 0U; wt < Wt; ++wt) {
      llk_wait_tiles(0,1);
      llk_unpack_reduce<PoolType::SUM,ReduceDim::REDUCE_ROW>(0,0);
      llk_pop_tiles(0,1);
    }
  }
}
}
}
