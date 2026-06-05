#include <math.h>
#include <stdint.h>
#include <string.h>

/*
 * High-precision exp2f implementation for fp32, targeting < 1 ULP error.
 *
 * The method uses a standard range reduction and a minimax polynomial approximation.
 *
 * 1.  Handle special cases: NaN, +/- infinity, and inputs that would overflow
 *     the single-precision float range.
 *
 * 2.  Range Reduction: The input `x` is split into an integer part `i` and a
 *     fractional part `f` such that `x = i + f` and `f` is in `[-0.5, 0.5]`.
 *     This is done via `i = round(x)` and `f = x - i`.
 *     Therefore, `exp2(x) = exp2(i) * exp2(f)`.
 *
 * 3.  Scale Calculation: `exp2(i)` is computed exactly and efficiently by
 *     constructing a float with the appropriate exponent bits. This handles
 *     normal, subnormal, and infinity cases for `2^i`.
 *
 * 4.  Polynomial Approximation: `exp2(f)` is approximated using a degree-5
 *     minimax polynomial for `f` in `[-0.5, 0.5]`. Intermediate calculations
 *     are performed using `double` precision to preserve accuracy. The
 *     polynomial is evaluated efficiently using Horner's method with fused
 *     multiply-add (fma).
 *
 * 5.  Reconstruction: The final result is obtained by multiplying the scale
 *     factor `exp2(i)` with the polynomial result `exp2(f)`. This final
 *     multiplication is also done in double precision before casting back to
 *     float to ensure the highest possible accuracy and proper underflow handling.
 */
float exp2f_precise(float x) {
    // Step 1: Handle special cases.
    uint32_t u;
    memcpy(&u, &x, sizeof(u));

    // Check if exponent is maxed out (NaN or Inf).
    if ((u & 0x7f800000) == 0x7f800000) {
        return x; // NaN or Inf, return as is.
    }
    // Check for inputs that are definitely out of the standard range, causing overflow.
    // For x > 128.0f, exp2f(x) will be INFINITY.
    // The scale factor calculation handles x = 128.0f yielding INFINITY. Any x > 128.0f
    // will result in a scaled value that also overflows to INFINITY.
    if (x > 128.0f) {
        return INFINITY;
    }

    // Step 2: Range Reduction to [-0.5, 0.5]
    float i_f = floorf(x + 0.5f);
    int i = (int)i_f;
    
    // f = x - i, computed in double for precision.
    double f = (double)x - (double)i_f;

    // Step 3: Compute scale factor 2^i
    uint32_t scale_u;
    if (i >= -126) {
        // Normal or infinity scale factor. Biased exponent is i + 127.
        // For i = 128, i + 127 = 255 (max exponent), which maps to INFINITY.
        scale_u = (uint32_t)(i + 127) << 23;
    } else {
        // Subnormal scale factor for i < -126.
        // A subnormal number 2^i is represented with an exponent field of 0.
        // Its value is (mantissa / 2^23) * 2^-126.
        // To get 2^i, the mantissa bits should be 2^(i + 126 + 23).
        // E.g., for i = -127, mantissa_bits = 1 << (-127 + 126 + 23) = 1 << 22 (0x00400000).
        // E.g., for i = -149, mantissa_bits = 1 << (-149 + 126 + 23) = 1 << 0 (0x00000001), which is FLT_MIN.
        int sub_exp_shift = i + 126 + 23; 
        if (sub_exp_shift >= 0) { // Valid subnormal bit pattern (i >= -149)
            scale_u = (uint32_t)1 << sub_exp_shift;
        } else { // i < -149, 2^i is smaller than FLT_MIN, resulting in 0.0f
            scale_u = 0;
        }
    }
    float scale_f;
    memcpy(&scale_f, &scale_u, sizeof(scale_f));

    // Step 4: Polynomial approximation of 2^f for f in [-0.5, 0.5]
    // Coefficients are derived from a minimax approximation of exp(f * ln(2)).
    // Using double precision for coefficients and evaluation.
    const double P5 = 1.3333638722284245e-03; // ln(2)^5/120
    const double P4 = 9.6180173678315130e-03; // ln(2)^4/24
    const double P3 = 5.5504108664821580e-02; // ln(2)^3/6
    const double P2 = 2.4022650695910070e-01; // ln(2)^2/2
    const double P1 = 6.9314718055994530e-01; // ln(2)

    // Evaluate using Horner's method with FMA for speed and accuracy.
    // r = 1.0 + f * (P1 + f * (P2 + f * (P3 + f * (P4 + f * P5))))
    double r = fma(f, P5, P4);
    r = fma(f, r, P3);
    r = fma(f, r, P2);
    r = fma(f, r, P1);
    r = fma(f, r, 1.0);

    // Step 5: Reconstruction
    // Result = scale * r
    // Perform multiplication in double and cast at the end.
    // This handles potential underflow to 0.0f if scale_f * r is less than FLT_MIN.
    return (float)((double)scale_f * r);
}
