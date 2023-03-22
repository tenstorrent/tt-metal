// Sfpu golden functions
float exponential(float x) {
    return exp(x);
}

float reciprocal(float x) {
    return 1 / x;
}

float gelu(float x) {
    static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
    auto x3 = x*x*x;
    return x*0.5*( 1.0 + tanhf( alpha * (x + 0.044715*x3) ) );
}

float relu(float x) {
    return fmaxf(x, 0.0f);
}

float ref_sqrt(float x) {
    return sqrtf(x);
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float ref_log(float x) {
    return logf(x);
}

float ref_tanh(float x) {
    return tanh(x);
}

vector<uint32_t> sfpu(const vector<uint32_t> &src, std::function<float(float)> sfpu_func) {
    vector<uint32_t> dst;

    for (uint32_t el: src) {

        uint32_t top = el & 0xffff0000;
        uint32_t bottom = el << 16;

        float top_ = *reinterpret_cast<float*>(&top);
        float bottom_ = *reinterpret_cast<float*>(&bottom);

        float exp_top = sfpu_func(top_);
        float exp_bottom = sfpu_func(bottom_);

        bfloat16 bfp16_top = bfloat16(exp_top);
        bfloat16 bfp16_bottom = bfloat16(exp_bottom);

        uint32_t new_val = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfp16_bottom, bfp16_top));
        dst.push_back(new_val);
    }
    return dst;
}

// Helper functions
vector<uint32_t> create_random_ones_and_twos_vector_of_bfloat16(uint32_t num_bytes, int seed) {
    // Used for reciprocal, since binary vectors are filled with 0s and 1s, and recip of 0 is undefined,
    // so then we just generate a vector of ones and twos

    vector<uint32_t> src = create_random_binary_vector_of_bfloat16(num_bytes, seed);

    vector<uint32_t> dst;

    for (uint32_t el: src) {

        uint32_t top = el & 0xffff0000;
        uint32_t bottom = el << 16;

        float top_ = *reinterpret_cast<float*>(&top);
        float bottom_ = *reinterpret_cast<float*>(&bottom);

        float top_plus_one = 1 + top_;
        float bottom_plus_one = 1 + bottom_;

        bfloat16 bfp16_top = bfloat16(top_plus_one);
        bfloat16 bfp16_bottom = bfloat16(bottom_plus_one);

        uint32_t new_val = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfp16_bottom, bfp16_top));
        dst.push_back(new_val);
    }
    return dst;
}

// Comparison functions
bool equal_within_two_sig_figs(float a, float b) {
    return equal_within_n_sig_figs(a, b, 2);
}

bool equal_within_absolute_tolerance_of_0p03(float a, float b) {
    return equal_within_absolute_tolerance(a, b, 0.03);
}

bool is_close_0p015(float a, float b) {
    return is_close(a, b, 0.015f);
}

bool is_close_rtol_0p06_atol_0p006(float a, float b) {
    return is_close(a, b, 0.06f, 0.006f);
}

bool is_close_rtol_0p175_atol_0p1(float a, float b) {
    return is_close(a, b, 0.175f, 0.1f);
}

// SFPU maps -> relevant kernels, golden functions, comparison functions
const map<string, string> sfpu_op_to_hlk_op_name = {
    // TODO(AP): auto-generate inits
    { "relu", "pack_relu_tile_to_stream(0, CB::c_out0);" },
    { "exponential", "exp_tile_init(); exp_tile(0); pack_tile(0, CB::c_out0);" },
    { "reciprocal", "recip_tile_init(); recip_tile(0); pack_tile(0, CB::c_out0);" },
    { "gelu", "gelu_tile_init(); gelu_tile(0); pack_tile(0, CB::c_out0);" },
    { "sqrt", "sqrt_tile_init(); sqrt_tile(0); pack_tile(0, CB::c_out0);" },
    { "sigmoid", "sigmoid_tile_init(); sigmoid_tile(0); pack_tile(0, CB::c_out0);" },
    { "log", "log_tile_init(); log_tile(0); pack_tile(0, CB::c_out0);" },
    { "tanh", "tanh_tile_init(); tanh_tile(0); pack_tile(0, CB::c_out0);" },
};

const map<string, std::function<float(float)>> sfpu_op_to_function = {
    {"relu",        relu},
    {"exponential", exponential},
    {"reciprocal",  reciprocal},
    {"gelu",        gelu},
    {"sqrt",        ref_sqrt},
    {"sigmoid",     sigmoid},
    {"log",         ref_log},
    {"tanh",        ref_tanh},
};

const map<string, std::function<vector<uint32_t>(uint32_t num_bytes, int seed)>> sfpu_op_to_init_func = {
    {"relu",        create_random_vector_of_bfloat16_1_1},
    {"exponential", create_random_binary_vector_of_bfloat16},
    {"reciprocal",  create_random_ones_and_twos_vector_of_bfloat16},
    {"gelu",        create_random_binary_vector_of_bfloat16},
    {"sqrt",        create_random_vector_of_bfloat16_0_2},
    {"sigmoid",     create_random_vector_of_bfloat16_1_1},
    {"log",         create_random_vector_of_bfloat16_0_2},
    {"tanh",        create_random_vector_of_bfloat16_1_1},
};

const map<string, std::function<bool(float a, float b)>> sfpu_op_to_comparison_function = {
    {"exponential", equal_within_two_sig_figs},
    {"reciprocal", equal_within_absolute_tolerance_of_0p03},
    {"gelu", is_close_0p015},
    {"relu", is_close_0p015},
    {"sqrt", is_close_rtol_0p06_atol_0p006},
    {"sigmoid", is_close_rtol_0p06_atol_0p006},
    {"log", is_close_rtol_0p06_atol_0p006},
    {"tanh", is_close_rtol_0p175_atol_0p1},
};
