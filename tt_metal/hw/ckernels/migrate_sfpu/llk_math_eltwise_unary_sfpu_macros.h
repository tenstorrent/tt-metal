// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
 
// ##############
//  MAIN MACROS
//  #############

// Overall approach: llk_math_eltwise_unary_sfpu_* functions follow a common structure
// of wrapping a single function call, but they differ in the exact template and function parameters
// used. Therefore, we use a set of macro helpers that can handle declaring arbitrary parameter lists
// and passing them through to a function call as arguments (optionally with some additional literal values).

// PARAM(type, name): Define a parameter
#define PARAM(type, name) (type, name, )
// DEFAULT_PARAM(type, name, val): Define a parameter with a default
#define DEFAULT_PARAM(type, name, val) (type, name, = val)
// ARG(val): Define an argument that will be left out of parameter declarations but included in usage lists
#define ARG(val) (, , val)

// Group multiple parameters together
#define PARAM_LIST(...) __VA_ARGS__

// Define an init function
// Variants:
//   - SfpuType::* might mismatch the op name
//   - llk_math_eltwise_unary_sfpu_init may be passed an initialization function (OP_init by default),
//     which may involve some additional pass-through parameters
#define SFPU_INIT(OP) SFPU_INIT_MANUAL(OP, OP)
#define SFPU_INIT_CUSTOM_NAME(OP, OP_TYPE_NAME) SFPU_INIT_MANUAL(OP, OP_TYPE_NAME)
#define SFPU_INIT_WITH_FN(OP, ...) \
    SFPU_INIT_MANUAL(OP, OP, ARG(sfpu::OP##_init<APPROXIMATE>) __VA_OPT__(, ) __VA_ARGS__)
#define SFPU_INIT_CUSTOM_NAME_WITH_FN(OP, OP_TYPE_NAME, FN_NAME, ...) \
    SFPU_INIT_MANUAL(OP, OP_TYPE_NAME, ARG(sfpu::FN_NAME<APPROXIMATE>) __VA_OPT__(, ) __VA_ARGS__)

// General init function template. Should not need to get called directly
#define SFPU_INIT_MANUAL(OP, OP_TYPE_NAME, ...)                                                                       \
    template <bool APPROXIMATE>                                                                                       \
    inline void llk_math_eltwise_unary_sfpu_##OP##_init(REMOVE_COMMA(DECL_PARAMS(__VA_ARGS__))) {                     \
        llk_math_eltwise_unary_sfpu_init<SfpuType::OP_TYPE_NAME, APPROXIMATE>(REMOVE_COMMA(USE_PARAMS(__VA_ARGS__))); \
    }

// Define a calculate function (with optional parameter list)
#define SFPU_CALCULATE(OP, ...)                               \
    SFPU_CALCULATE_MANUAL(                                    \
        OP,                                                   \
        calculate_##OP,                                       \
        DEFAULT_PARAM(int, vector_mode, (int)VectorMode::RC), \
        PARAM_LIST(),                                         \
        PARAM_LIST(__VA_ARGS__))

// Calculate functions, split by the vector_mode parameter
// These allow non-standard ckernel names and adding template parameters/arguments
#define SFPU_CALCULATE_RC(OP, CKERNEL_NAME, TEMPLATE_PARAMS, FN_PARAMS) \
    SFPU_CALCULATE_MANUAL(                                              \
        OP,                                                             \
        CKERNEL_NAME,                                                   \
        DEFAULT_PARAM(int, vector_mode, (int)VectorMode::RC),           \
        PARAM_LIST(TEMPLATE_PARAMS),                                    \
        PARAM_LIST(FN_PARAMS))
#define SFPU_CALCULATE_ALWAYS_RC(OP, CKERNEL_NAME, TEMPLATE_PARAMS, FN_PARAMS) \
    SFPU_CALCULATE_MANUAL(                                                     \
        OP, CKERNEL_NAME, ARG((int)VectorMode::RC), PARAM_LIST(TEMPLATE_PARAMS), PARAM_LIST(FN_PARAMS))
#define SFPU_CALCULATE_RC_CUSTOM(OP, CKERNEL_NAME, TEMPLATE_PARAMS, FN_PARAMS) \
    SFPU_CALCULATE_MANUAL(                                                     \
        OP,                                                                    \
        CKERNEL_NAME,                                                          \
        DEFAULT_PARAM(int, vector_mode, (int)VectorMode::RC_custom),           \
        PARAM_LIST(TEMPLATE_PARAMS),                                           \
        PARAM_LIST(FN_PARAMS))
// Calculate functions where the APPROXIMATE template argument is passed on in the last position rather than the first
#define SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(OP, CKERNEL_NAME, TEMPLATE_PARAMS, FN_PARAMS) \
    SFPU_CALCULATE_MANUAL_APPROX_LAST(OP, CKERNEL_NAME, ARG((int)VectorMode::RC), TEMPLATE_PARAMS, FN_PARAMS)
#define SFPU_CALCULATE_RC_APPROX_LAST(OP, CKERNEL_NAME, TEMPLATE_PARAMS, FN_PARAMS) \
    SFPU_CALCULATE_MANUAL_APPROX_LAST(                                              \
        OP, CKERNEL_NAME, DEFAULT_PARAM(int, vector_mode, (int)VectorMode::RC), TEMPLATE_PARAMS, FN_PARAMS)

#define SFPU_CALCULATE_MANUAL(OP, CKERNEL_NAME, VEC_MODE, TEMPLATE_PARAMS, FN_PARAMS)                           \
    template <bool APPROXIMATE DECL_PARAMS(TEMPLATE_PARAMS)>                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index DECL_PARAMS(FN_PARAMS) DECL_PARAMS(VEC_MODE)) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                        \
            ckernel::sfpu::CKERNEL_NAME<APPROXIMATE USE_PARAMS(TEMPLATE_PARAMS)>,                               \
            dst_index USE_PARAMS(VEC_MODE) USE_PARAMS(FN_PARAMS));                                              \
    }

#define SFPU_CALCULATE_MANUAL_APPROX_LAST(OP, CKERNEL_NAME, VEC_MODE, TEMPLATE_PARAMS, FN_PARAMS)               \
    template <bool APPROXIMATE DECL_PARAMS(TEMPLATE_PARAMS)>                                                    \
    inline void llk_math_eltwise_unary_sfpu_##OP(uint dst_index DECL_PARAMS(FN_PARAMS) DECL_PARAMS(VEC_MODE)) { \
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                                                        \
            ckernel::sfpu::CKERNEL_NAME<REMOVE_COMMA(TRAIL_COMMA(USE_PARAMS(TEMPLATE_PARAMS))) APPROXIMATE>,    \
            dst_index USE_PARAMS(VEC_MODE) USE_PARAMS(FN_PARAMS));                                              \
    }

// ##################
//  SUPPORTING MACROS
// ##################

// DECL_PARAMS(param1, param2, ...): Declare a list of parameters in the function signature
// Params should be created by PARAM, DEFAULT_PARAM, or ARG
// Returned text will have a leading comma
#define DECL_PARAMS(...) FOR_EACH(DECL_PARAM_SINGLE, __VA_ARGS__)
#define DECL_PARAM_SINGLE(param) DECL_PARAM_IMPL param
#define DECL_PARAM_IMPL(type, name, val) IF_EMPTY(name)(, ADD_COMMA(type name val))
#define ADD_COMMA(x) , x

// USE_PARAMS(param1, param2, ...): Pass a list of parameters to another function
// Params should be created by PARAM, DEFAULT_PARAM, or ARG
// Returned text will have a leading comma
#define USE_PARAMS(...) FOR_EACH(USE_PARAM_SINGLE, __VA_ARGS__)
#define USE_PARAM_SINGLE(param) USE_PARAM_IMPL param
#define USE_PARAM_IMPL(type, name, val) , IF_EMPTY(name)(val, name)
// COUNT_ARGS(...): Count variable macro arguments
// (Supports up to 10 args)
#define COUNT_ARGS(...) COUNT_ARGS_IMPL(__VA_ARGS__ __VA_OPT__(, ) 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

// FOR_EACH(F, ...): Apply a macro `F` to all of its arguments
// (Supports up to 10 args)
#define FOR_EACH(F, ...) FOR_EACH_IMPL(F, COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define FOR_EACH_IMPL(F, N, ...) FOR_EACH_N(F, N, __VA_ARGS__)
#define FOR_EACH_N(F, N, ...) FOR_EACH_##N(F, __VA_ARGS__)

#define FOR_EACH_0(F, ...)
#define FOR_EACH_1(F, p1) F(p1)
#define FOR_EACH_2(F, p1, p2) F(p1) F(p2)
#define FOR_EACH_3(F, p1, p2, p3) F(p1) F(p2) F(p3)
#define FOR_EACH_4(F, p1, p2, p3, p4) F(p1) F(p2) F(p3) F(p4)
#define FOR_EACH_5(F, p1, p2, p3, p4, p5) F(p1) F(p2) F(p3) F(p4) F(p5)
#define FOR_EACH_6(F, p1, p2, p3, p4, p5, p6) F(p1) F(p2) F(p3) F(p4) F(p5) F(p6)
#define FOR_EACH_7(F, p1, p2, p3, p4, p5, p6, p7) F(p1) F(p2) F(p3) F(p4) F(p5) F(p6) F(p7)
#define FOR_EACH_8(F, p1, p2, p3, p4, p5, p6, p7, p8) F(p1) F(p2) F(p3) F(p4) F(p5) F(p6) F(p7) F(p8)
#define FOR_EACH_9(F, p1, p2, p3, p4, p5, p6, p7, p8, p9) F(p1) F(p2) F(p3) F(p4) F(p5) F(p6) F(p7) F(p8) F(p9)
#define FOR_EACH_10(F, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10) \
    F(p1) F(p2) F(p3) F(p4) F(p5) F(p6) F(p7) F(p8) F(p9) F(p10)

// IF_EMPTY(arg)(YES, NO): Conditional switch based on whether first argument is empty or not
// The odd structure of returning an argument selector macro is required to make the comma arguments
// work correctly in DECL_PARAMS and USE_PARAMS
#define IF_EMPTY(...) GET##__VA_OPT__(2)
#define GET(a, b) a
#define GET2(a, b) b

// Remove leading comma from the argument list
#define REMOVE_COMMA(...) REMOVE_COMMA_IMPL(__VA_ARGS__)
#define REMOVE_COMMA_IMPL(x, ...) __VA_ARGS__

// Add a trailing comma for non-empty argument lists
#define TRAIL_COMMA(...) __VA_ARGS__ __VA_OPT__(, )
