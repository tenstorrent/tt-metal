sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {
    sfpi::vFloat result = 0.f;
    v_if(val == val) { result = 3.f; }
    v_else { result = -3.f; }
    v_endif;

    return result;
}

void calculate_sfpi_kernel_init() {}
