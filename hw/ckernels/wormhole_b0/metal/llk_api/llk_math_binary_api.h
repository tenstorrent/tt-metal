
template<PoolType type, ReduceDim dim, DestAccumulation fp32_dest_accumulation, int MATH_FIDELITY_DESC = 0>
inline void _llk_math_reduce_(const uint dst_index) {
    if (fp32_dest_accumulation == DestAccumulation::Enable) {
        // do something
    }
}
