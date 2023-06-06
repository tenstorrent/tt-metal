#pragma once
#include <cstdint>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;

/*
* This file contains values that are visible to both host and device compiled code.
*/

constexpr static std::uint32_t INVALID = 0;
constexpr static std::uint32_t VALID = 1;
constexpr static std::uint32_t NOTIFY_HOST_KERNEL_COMPLETE_VALUE = 512;
