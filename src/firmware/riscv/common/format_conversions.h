#include <cstddef>
#include <cstdint>

/** @brief Convert fp16 to uint16, dropping all fractional bits
 *
 * does not handle negatives, infs or NaNs
 */
inline std::uint16_t fp16_to_unsigned_index(std::uint16_t fp16) {
  constexpr int kFractionBits = 10;
  constexpr std::uint16_t kMantissaMask = (1 << kFractionBits) - 1;
  std::uint16_t offset_exponent = (fp16 & 0b111'1100'0000'0000) >> kFractionBits;
  std::uint32_t mantissa = (1 << kFractionBits) | (fp16 & kMantissaMask);
  if (offset_exponent == 0) {
    return 0;  // handle 0. Subnormals also trigger this
  }
  std::uint16_t exponent = offset_exponent - 15;
  std::uint32_t fixed_point = mantissa << exponent;
  return fixed_point >> kFractionBits;  // drop the fractional bits
}

/** @brief Convert brain float 16 to uint32, dropping all fractional bits
 *
 * does not handle negatives, infs or NaNs
 */
inline std::uint32_t fp16_brain_to_unsigned_index(std::uint16_t fp16) {
  constexpr int kFractionBits = 7;
  constexpr std::uint16_t kMantissaMask = (1 << kFractionBits) - 1;
  std::uint32_t offset_exponent = (fp16 & 0b1111'1111'000'0000) >> kFractionBits;
  std::uint32_t mantissa = (1 << kFractionBits) | (fp16 & kMantissaMask);
  if (offset_exponent == 0) {
    return 0;  // handle 0. Subnormals also trigger this
  }
  std::uint32_t exponent = offset_exponent - 127;
  std::uint64_t fixed_point = (std::uint64_t)mantissa << exponent;
  return fixed_point >> kFractionBits;  // drop the fractional bits
}
