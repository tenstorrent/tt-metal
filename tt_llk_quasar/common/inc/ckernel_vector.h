// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Disabled until we can update GCC with RISC-V vector extension
#define HAVE_RISCV_VECTOR 1

#ifdef HAVE_RISCV_VECTOR

// First-class enum types to get smarter template type deduction

enum vreg
{
    V0 = 0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    V11,
    V12,
    V13,
    V14,
    V15,
    V16,
    V17,
    V18,
    V19,
    V20,
    V21,
    V22,
    V23,
    V24,
    V25,
    V26,
    V27,
    V28,
    V29,
    V30,
    V31
};

enum vdatasz
{
    E8 = 0,
    E16,
    E32,
    E64,
    NUM_VDATA_SIZES
};

uint32_t constexpr vdatasz_encodings[NUM_VDATA_SIZES] = {0, 1, 2, 3};
uint32_t constexpr vdatasz_asm_nums[NUM_VDATA_SIZES]  = {8, 16, 32, 64};

template <typename T>
struct _type_to_datasz
{
};

template <>
struct _type_to_datasz<float>
{
    static vdatasz const sz = E32;
};

template <>
struct _type_to_datasz<uint32_t>
{
    static vdatasz const sz = E32;
};

template <>
struct _type_to_datasz<int32_t>
{
    static vdatasz const sz = E32;
};

template <>
struct _type_to_datasz<uint16_t>
{
    static vdatasz const sz = E16;
};

template <>
struct _type_to_datasz<int16_t>
{
    static vdatasz const sz = E16;
};

template <>
struct _type_to_datasz<uint8_t>
{
    static vdatasz const sz = E8;
};

template <>
struct _type_to_datasz<int8_t>
{
    static vdatasz const sz = E8;
};

#define type_to_datasz(T) (_type_to_datasz<T>::sz)

// MF = multiplier (fractional). For example, MF4 represents 1/4
enum vregmult
{
    MF8 = 0,
    MF4,
    MF2,
    M1,
    M2,
    M4,
    M8,
    NUM_VREGMULTS
};

uint32_t constexpr vregmult_encodings[NUM_VREGMULTS] = {5, 6, 7, 0, 1, 2, 3};
uint32_t constexpr vregmult_asm_nums[NUM_VREGMULTS]  = {8, 4, 2, 1, 2, 4, 8};

template <vdatasz SEW, vregmult LMUL = M1>
inline uint32_t vsetvl(uint32_t appn_vec_len)
{
    uint32_t chunk_sz;
    if (LMUL < M1)
    {
        uint32_t constexpr e_val = vdatasz_asm_nums[SEW];
        uint32_t constexpr m_val = vregmult_asm_nums[LMUL];
        (void)e_val; // Prevent unused variable error
        (void)m_val; // Prevent unused variable error
        asm volatile("vsetvli %[chunk_sz], %[avl], e%c[e_val], mf%c[m_val] \n"
                     : [chunk_sz] "=r"(chunk_sz)
                     : [avl] "r"(appn_vec_len), [e_val] "i"(e_val), [m_val] "i"(m_val));
    }
    else
    {
        uint32_t constexpr e_val = vdatasz_asm_nums[SEW];
        uint32_t constexpr m_val = vregmult_asm_nums[LMUL];
        (void)e_val; // Prevent unused variable error
        (void)m_val; // Prevent unused variable error
        asm volatile("vsetvli %[chunk_sz], %[avl], e%c[e_val], m%c[m_val] \n"
                     : [chunk_sz] "=r"(chunk_sz)
                     : [avl] "r"(appn_vec_len), [e_val] "i"(e_val), [m_val] "i"(m_val));
    }

    return chunk_sz;
}

template <vdatasz SEW, vregmult LMUL, uint32_t appn_vec_len>
inline uint32_t vsetvl()
{
    if (appn_vec_len > 31)
    {
        // Given application vector length is too big to fit in the 5-bit (unsigned) immediate
        return vsetvl<SEW, LMUL>(appn_vec_len);
    }

    uint32_t chunk_sz;
    if (LMUL < M1)
    {
        uint32_t constexpr e_val = vdatasz_asm_nums[SEW];
        uint32_t constexpr m_val = vregmult_asm_nums[LMUL];
        (void)e_val; // Prevent unused variable error
        (void)m_val; // Prevent unused variable error
        asm volatile("vsetivli %[chunk_sz], %[avl], e%c[e_val], mf%c[m_val] \n"
                     : [chunk_sz] "=r"(chunk_sz)
                     : [avl] "i"(appn_vec_len), [e_val] "i"(e_val), [m_val] "i"(m_val));
    }
    else
    {
        uint32_t constexpr e_val = vdatasz_asm_nums[SEW];
        uint32_t constexpr m_val = vregmult_asm_nums[LMUL];
        (void)e_val; // Prevent unused variable error
        (void)m_val; // Prevent unused variable error
        asm volatile("vsetivli %[chunk_sz], %[avl], e%c[e_val], m%c[m_val] \n"
                     : [chunk_sz] "=r"(chunk_sz)
                     : [avl] "i"(appn_vec_len), [e_val] "i"(e_val), [m_val] "i"(m_val));
    }

    return chunk_sz;
}

template <vreg vec_reg_no, typename T, vdatasz data_size = type_to_datasz(T)>
inline void vector_load(T const *addr)
{
    uint32_t constexpr e_val = vdatasz_asm_nums[data_size];
    asm volatile("vle%c[e_val].v v%c[vec_reg_no], (%[addr]) \n" : : [e_val] "i"(e_val), [vec_reg_no] "i"(vec_reg_no), [addr] "r"(addr));
}

// For backwards compatibility
template <vdatasz data_size, vreg vec_reg_no, typename T>
inline void vector_load(T const *addr)
{
    uint32_t constexpr e_val = vdatasz_asm_nums[data_size];
    asm volatile("vle%c[e_val].v v%c[vec_reg_no], (%[addr]) \n" : : [e_val] "i"(e_val), [vec_reg_no] "i"(vec_reg_no), [addr] "r"(addr));
}

template <vreg vec_reg_no, typename T, vdatasz data_size = type_to_datasz(T)>
inline void vector_store(T *addr)
{
    uint32_t constexpr e_val = vdatasz_asm_nums[data_size];
    asm volatile("vse%c[e_val].v v%c[vec_reg_no], (%[addr]) \n" : : [e_val] "i"(e_val), [vec_reg_no] "i"(vec_reg_no), [addr] "r"(addr));
}

// For backwards compatibility
template <vdatasz data_size, vreg vec_reg_no, typename T>
inline void vector_store(T *addr)
{
    uint32_t constexpr e_val = vdatasz_asm_nums[data_size];
    asm volatile("vse%c[e_val].v v%c[vec_reg_no], (%[addr]) \n" : : [e_val] "i"(e_val), [vec_reg_no] "i"(vec_reg_no), [addr] "r"(addr));
}

template <vreg dest_vec_reg_no, vreg src_vec_reg_no, int shift_amt>
inline void vector_slide_down()
{
    asm volatile("vslidedown.vi v%c[dst], v%c[src], %[amt] \n" : : [dst] "i"(dest_vec_reg_no), [src] "i"(src_vec_reg_no), [amt] "i"(shift_amt));
}

template <vreg dest_vec_reg_no, vreg src_vec_reg_no>
inline void vector_slide_down(int shift_amt)
{
    asm volatile("vslidedown.vx v%c[dst], v%c[src], %[amt] \n" : : [dst] "i"(dest_vec_reg_no), [src] "i"(src_vec_reg_no), [amt] "r"(shift_amt));
}

template <vreg vec_reg_no>
inline uint32_t vector_pop_front()
{
    uint32_t val;
    asm volatile("vmv.x.s %[val], v%c[src]\n" : [val] "=r"(val) : [src] "i"(vec_reg_no));

    return val;
}

// Fills vector with 0,1,2,...
template <vreg vec_reg_no>
inline void vector_iota()
{
    asm volatile("vid.v v%c[dst]\n" : : [dst] "i"(vec_reg_no));
}

#define mk_vector_binary_op(op)                                                                                                                     \
    template <vreg dest_vec_reg_no, vreg src1_vec_reg_no, vreg src2_vec_reg_no>                                                                     \
    inline void vector_##op()                                                                                                                       \
    {                                                                                                                                               \
        asm volatile("v" #op ".vv v%c[dst], v%c[src1], v%c[src2] \n"                                                                                \
                     :                                                                                                                              \
                     : [dst] "i"(dest_vec_reg_no), [src1] "i"(src1_vec_reg_no), [src2] "i"(src2_vec_reg_no));                                       \
    }                                                                                                                                               \
    template <vreg dest_vec_reg_no, vreg src1_vec_reg_no>                                                                                           \
    inline void vector_##op(int32_t val)                                                                                                            \
    {                                                                                                                                               \
        asm volatile("v" #op ".vx v%c[dst], v%c[src1], %[val] \n" : : [dst] "i"(dest_vec_reg_no), [src1] "i"(src1_vec_reg_no), [val] "r"(val));     \
    }                                                                                                                                               \
    template <vreg dest_vec_reg_no, vreg src1_vec_reg_no, int32_t imm>                                                                              \
    inline void vector_##op()                                                                                                                       \
    {                                                                                                                                               \
        if (imm < -16 || imm > 15)                                                                                                                  \
        {                                                                                                                                           \
            /*Immediate too big; just use the register version*/                                                                                    \
            vector_##op<dest_vec_reg_no, src1_vec_reg_no>(static_cast<int32_t>(imm));                                                               \
        }                                                                                                                                           \
        else                                                                                                                                        \
        {                                                                                                                                           \
            asm volatile("v" #op ".vi v%c[dst], v%c[src1], %[imm] \n" : : [dst] "i"(dest_vec_reg_no), [src1] "i"(src1_vec_reg_no), [imm] "i"(imm)); \
        }                                                                                                                                           \
    }                                                                                                                                               \
    struct swallow_semicolon_##op                                                                                                                   \
    {                                                                                                                                               \
    }

mk_vector_binary_op(add);

#endif
