/**
 * @file
 * @brief Macros and utility for using the FWLOG feature of versim, emule & chip.
 *
 * NOTE: Strings must be literals, and macro usage must not have any '"' characters, except to delimit the string.
 *
 * This header defines several macros for logging from firmware that will show up in versim
 *   and emule output, while minimally impacting cycle counts.
 * The usual macros follow. All of them ensure any side-effects of the non-string arguments happen,
 *   even if FWLOG is not enabled.
 * For macros that take non-string arguments, the same macro is available with no side-effects by appending @c _NSE .
 * If you want to keep timings as similar to when FWLOG is not enabled, but still want it to print, append @c _ALWAYS .
 * The @c _ALWAYS macros are for performance testing and debugging timing-related issues, and should not be committed.
 *
 * FWASSERT(str, cont)
 *   If the contition is false, causes emule/versim to print the message stop the simulation
 * FWLOG0(str)
 * FWLOG1(fmt, a)
 * FWLOG2(fmt, a, b)
 *   Print a message taking up to two arguments. @p a and @p b are always passed as @c uint32_t
 * FWEVENT(str)
 *   Print the message using the separate FWEVENT machinery
 *
 * Using firmware with FWLOG enabled can significantly increase runtime, resulting in timing changes.
 * On average, 0-argument ones (FWLOG0, FWEVENT) take 5 cycles, 1-argument ones (FWLOG1, FWASSERT) take 10 cycles,
 *   and 2-argument ones (FWLOG2) take 15 cycles. (2020-05-29)
 * Relatedly, reading the wall clock and subtracting from a start time adds about 4 cycles
 * FWLOG enabled also increase image size, up to about 2KiB for main firmware.
 *
 * This file is tightly coupled with fwlog.py, which parses the source code to extract the format strings.
 * It just uses a regex, which results in the above note.
 */
#pragma once

#include <cstdint>

// Constants definining usage of the bits in the 'locator'. Must match fwlog.py
const int NUM_BITS_FOR_FORMAT_HASH = 12;
const int NUM_BITS_FOR_LINE_NUM = 32 - NUM_BITS_FOR_FORMAT_HASH;

// This function must match python function in fwlog.py
uint32_t constexpr fwlog_string_hash(const char *f) {
    uint32_t h = 5381;
    for (int i = 0; f[i] != 0; i++) {
      h = h*33 + (uint32_t)f[i];
    }
    h = h & ((1 << NUM_BITS_FOR_FORMAT_HASH) - 1);
    return h;
}

uint32_t constexpr fwlog_hash_combine(std::uint32_t format_hash, std::uint32_t line) {
  return ((format_hash) << NUM_BITS_FOR_LINE_NUM) | line;
}

// using this should get picked up by fwlog.py too
#define FWLOG_HASH(fmt) fwlog_hash_combine(fwlog_string_hash(fmt), (std::uint32_t)__LINE__)

extern volatile uint32_t FWLOG_MAILBOX;
extern volatile uint32_t FWEVENT_MAILBOX;

// use template param to ensure compile-time computation
// these won't cause linker errors in emule unless actually called directly, which they shouldn't ever be
template <std::uint32_t locator>
void fwlog0() {
  volatile std::uint32_t *logbox = &FWLOG_MAILBOX;
  logbox[0] = locator;
}

template <std::uint32_t locator>
void fwlog1(std::uint32_t a) {
  volatile std::uint32_t *logbox = &FWLOG_MAILBOX;
  logbox[2] = a;
  logbox[0] = locator;
}

template <std::uint32_t locator>
void fwlog2(std::uint32_t a, std::uint32_t b) {
  volatile std::uint32_t *logbox = &FWLOG_MAILBOX;
  logbox[2] = a;
  logbox[3] = b;
  logbox[0] = locator;
}





/******************************
 * Define implementation macros
 ******************************/

#ifdef TENSIX_FIRMWARE
// versim/chip case

// FWASSERT looks like just a conditional log, because that's what it is
// (fwlog.py tells versim that a log with this locator is a FWASSERT)
#define FWASSERT_IMPL(fmt, cond) if (!(cond)) fwlog0<FWLOG_HASH(fmt)>()
#define FWEVENT_IMPL(fmt) *(&FWEVENT_MAILBOX) = FWLOG_HASH(fmt)
#define FWLOG0_IMPL(fmt) fwlog0<FWLOG_HASH(fmt)>()
#define FWLOG1_IMPL(fmt, a) fwlog1<FWLOG_HASH(fmt)>(a)
#define FWLOG2_IMPL(fmt, a, b) fwlog2<FWLOG_HASH(fmt)>(a, b)
#define FWLOG1STR_EMULE_IMPL(fmt, a) (void)((uint32_t)(a))

#else  // TENSIX_FIRMWARE
// emule/x86 case

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define FWASSERT_IMPL(fmt, cond) do { if (!(cond)) core->fwassertfail(__FILE__, __LINE__, fmt, #cond); } while (0)
#define FWEVENT_IMPL(fmt) do { core->fwlog(__FILE__, __LINE__, fmt, 0, 0); } while (0)
#define FWLOG0_IMPL(fmt) do { core->fwlog(__FILE__, __LINE__, fmt, 0, 0); } while (0)
#define FWLOG1_IMPL(fmt, a) do { core->fwlog(__FILE__, __LINE__, fmt, a, 0); } while (0)
#define FWLOG2_IMPL(fmt, a, b) do { core->fwlog(__FILE__, __LINE__, fmt, a, b); } while (0)
#define FWLOG1STR_EMULE_IMPL(fmt, a) do { core->fwlogstr(__FILE__, __LINE__, fmt, a); } while (0)

#endif // TENSIX_FIRMWARE





/******************************
 * define the public API macros
 ******************************/

#ifdef ENABLE_FWLOG

#define FWASSERT FWASSERT_IMPL
#define FWEVENT FWEVENT_IMPL
#define FWLOG0 FWLOG0_IMPL
#define FWLOG1 FWLOG1_IMPL
#define FWLOG2 FWLOG2_IMPL
#define FWLOG1STR_EMULE FWLOG1STR_EMULE_IMPL

// no side-effects versions. You'll have to do your own void casting if needed
#define FWASSERT_NSE FWASSERT
#define FWLOG1_NSE FWLOG1
#define FWLOG2_NSE FWLOG2

#elif defined(ENABLE_FWASSERT) // ENABLE_FW_ASSERT
#define FWASSERT FWASSERT_IMPL
// Rest of these are set to nothing as below.
#define FWEVENT(fmt)
#define FWLOG0(fmt)
#define FWLOG1(fmt, a) (void)((uint32_t)(a))
#define FWLOG2(fmt, a, b) (void)((uint32_t)(a)); (void)((uint32_t)(b))
#define FWLOG1STR_EMULE(fmt, a)  // note: no side-effects!

// no side-effects versions. You'll have to do your own void casting if needed
#define FWASSERT_NSE(fmt, cond)
#define FWLOG1_NSE(fmt, a)
#define FWLOG2_NSE(fmt, a, b)

#else // ENABLE_FWLOG

// Stub everything out to nothing. Use void casts to suppress unused warnings & ensure side-effects.
// void cast only needed for a & b because fmt must be a string literal
// Not clear if side-effects should happen... eg. C's `assert` doesn't have them when NDEBUG is defined
#define FWASSERT(fmt, cond) (void)((bool)(cond))
#define FWEVENT(fmt)
#define FWLOG0(fmt)
#define FWLOG1(fmt, a) (void)((uint32_t)(a))
#define FWLOG2(fmt, a, b) (void)((uint32_t)(a)); (void)((uint32_t)(b))
#define FWLOG1STR_EMULE(fmt, a)  // note: no side-effects!

// no side-effects versions. You'll have to do your own void casting if needed
#define FWASSERT_NSE(fmt, cond)
#define FWLOG1_NSE(fmt, a)
#define FWLOG2_NSE(fmt, a, b)

#endif // ENABLE_FWLOG

// public API for when you want to use FWLOG, but not have any other FWLOG enabled
// useful for things like cycle counting
#define FWASSERT_ALWAYS FWASSERT_IMPL
#define FWEVENT_ALWAYS FWEVENT_IMPL
#define FWLOG0_ALWAYS FWLOG0_IMPL
#define FWLOG1_ALWAYS FWLOG1_IMPL
#define FWLOG2_ALWAYS FWLOG2_IMPL
#define FWLOG1STR_EMULE_ALWAYS FWLOG1STR_EMULE_IMPL
