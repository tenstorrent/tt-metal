#include "error.h"
#include "internal.h"
#include "options.h"
#include "print.h"

#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef NOPTIONS

static const opt table[] = {
#define OPTION(N,V,L,H,D,F) \
  { #N, (int)(V), D },
  OPTIONS
#undef OPTION
};

#else

static const opt table[] = {
#define OPTION(N,V,L,H,D,F) \
  { #N, (int)(V), (int)(L), (int)(H), D, (bool)(F) },
  OPTIONS
#undef OPTION
};

#endif

#define size_table (sizeof table / sizeof * table)

const opt *kissat_options_begin = table;
const opt *kissat_options_end = table + size_table;

static void check_table_sorted(void) {
#ifndef NDEBUG
  opt const *p = 0;
  for (all_options(o))
    if (p && strcmp(p->name, o->name) >= 0) {
      kissat_fatal("option '%s' before option '%s'", p->name, o->name);
    } else {
      p = o;
    }
#endif
}

const opt *kissat_options_has(const char *name) {
  size_t l = 0, m, r = size_table;
  int tmp;
  opt const *o;
  assert(l < r);
  while (l + 1 < r) {
    m = l + (r - l) / 2;
    tmp = strcmp(name, (o = table + m)->name);
    if (tmp < 0) {
      r = m;
    } else if (tmp > 0) {
      l = m;
    } else {
      return o;
    }
  }
  o = table + l;
  tmp = strcmp(o->name, name);
  return tmp ? 0 : o;
}

bool kissat_parse_option_value(const char *val_str, int *res_ptr) {
  if (!strcmp(val_str, "true")) {
    *res_ptr = 1;
    return true;
  }
  if (!strcmp(val_str, "false")) {
    *res_ptr = 0;
    return true;
  }
  int sign = 1;
  char const *p = val_str;
  int ch = *p++;
  if (ch == '-') {
    sign = -1;
    ch = *p++;
  }
  if (!isdigit(ch)) {	// at least one digit
    return false;
  }
  const unsigned max = -(unsigned) INT_MIN;
  unsigned res = ch - '0';
  while (isdigit((ch = *p++))) {
    if (max / 10 < res) {
      return false;
    }
    res *= 10;
    const unsigned digit = ch - '0';
    if (max - digit < res) {
      return false;
    }
    res += digit;
    if (!res) {
      return false;  // invalid '00'
    }
  }
  if (ch == 'e') {	// parse '13e5' etc.
    if (!isdigit((ch = *p++))) {	// at least one digit
      return false;
    }
    if (res) {
      if (*p) {	// exactly one digit
        return false;
      }
      const unsigned digit = ch - '0';
      for (unsigned i = 0; i < digit; i++) {
        if (max / 10 < res) {
          return false;
        }
        res *= 10;
      }
    } else {		// parse '0^123123123' etc.
      while (isdigit(ch = *p++))	// arbitrary many digits
        ;
      if (ch) {
        return false;
      }
    }
  } else if (ch == '^') {	// parse '2^11' etc.
    const unsigned base = res;
    if (!isdigit((ch = *p++))) {	// at least one digit
      return false;
    }
    unsigned exp = ch - '0';
    if (base < 2) {	// parse '0^123123123' etc.
      while (isdigit(ch = *p++))	// arbitrary many digits
        ;
      if (ch) {
        return false;
      }
    } else if (isdigit(ch = *p++)) {	// parse '2^30' etc.
      if (*p) {	// at most two digits
        return false;
      }
      exp *= 10;
      const unsigned digit = ch - '0';
      exp += digit;
      if (!exp) {	// '2^00' invalid
        return false;
      }
    } else if (ch) {
      return false;
    }
    if (exp)
      for (unsigned i = 1; i < exp; i++) {
        if (max / base < res) {
          return false;
        }
        res *= base;
      } else if (base) {
      res = 1;  // parse '3^0'
    } else {
      return false;  // '0^0' invalid
    }
  } else if (ch) {
    return false;
  }
  assert(res <= max);
  if (sign > 0 && res == max) {
    return false;
  }
  res *= sign;
  *res_ptr = res;
  return true;
}

const char *kissat_parse_option_name(const char *arg, const char *name) {
  if (arg[0] != '-' || arg[1] != '-') {
    return 0;
  }
  char const *p = arg + 2, *q = name;
  while (*p && *p == *q) {
    p++, q++;
  }
  if (*q) {
    return 0;
  }
  if (*p != '=') {
    return 0;
  }
  return p + 1;
}

#ifdef NOPTIONS

void kissat_init_options(void) {
  check_table_sorted();
}

int kissat_options_get(const char *name) {
  const opt *const o = kissat_options_has(name);
  return o ? o->value : 0;
}

#else

#include "format.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

static void kissat_printf_usage(const char *option, const char *fmt, ...) {
  va_list ap;
  printf("  %-26s ", option);
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  fputc('\n', stdout);
}

static void check_ranges(void) {
#define OPTION(N,V,L,H,D,F) \
do { \
  if ((int)(L) > (int)(H)) \
    kissat_fatal ("minimum '%d' of option '%s' above maximum '%d'", \
      (int)(L), #N, (int)(H)); \
  if ((int)(V) < (int)(L)) \
    kissat_fatal ("default value '%d' of option '%s' below minimum '%d'", \
      (int)(V), #N, (int)(L)); \
  if ((int)(V) > (int)(H)) \
    kissat_fatal ("default value '%d' of option '%s' above maximum '%d'", \
      (int)(V), #N, (int)(H)); \
} while (0);
  OPTIONS
#undef OPTION
}

static void check_name_length(void) {
#ifndef NDEBUG
#define OPTION(N,V,L,H,D,F) \
  if (strlen (#N) + 1 > kissat_options_max_name_buffer_size) \
    kissat_fatal ("option '%s' name length %zu " \
      "exceeds maximum name buffer size %zu", \
      #N, strlen (#N), kissat_options_max_name_buffer_size);
  OPTIONS
#undef OPTION
#endif
}

int kissat_options_get(const options *options, const char *name) {
  const int *const p =
        kissat_options_ref(options, kissat_options_has(name));
  return p ? *p : 0;
}

int kissat_options_set_opt(
    options *options, const opt *o, int value, bool fixed) {
  assert(kissat_options_begin <= o);
  assert(o < kissat_options_end);
  int *p = (int *) options + (o - table);
  int res = *p;
  if (o->fixed && !fixed) {
    return res;
  }
  if (value == res) {
    return res;
  }
  if (value < o->low) {
    value = o->low;
  }
  if (value > o->high) {
    value = o->high;
  }
  *p = value;
  return res;
}

int kissat_options_set(
      options *options, const char *name, int value, bool fixed) {
  const opt *const o = kissat_options_has(name);
  if (!o) {
    return 0;
  }
  return kissat_options_set_opt(options, o, value, fixed);
}

void kissat_init_options(options *options) {
  check_ranges();
  check_name_length();
  check_table_sorted();
#define OPTION(N,V,L,H,D,F) \
  assert ((L) <= (V)); \
  assert ((V) <= (H)); \
  options->N = (V);
  OPTIONS
#undef OPTION
}

#define FORMAT_OPTION_LIMIT(V) \
  (((V) == INT_MIN || (V) == INT_MAX) ? \
    "." : kissat_format_value (&format, false, (V)))

void kissat_options_usage(kissat *solver) {
  check_ranges();
  check_name_length();
  check_table_sorted();
  format format;
  memset(&format, 0, sizeof format);
#define OPTION(N,V,L,H,D,F) \
  do { \
    const bool b = ((L) == 0 && (H) == 1); \
    char buffer[96]; \
    if (b) { \
      sprintf(buffer, "--%s=<bool>", #N); \
    } else { \
      const char *low_str = FORMAT_OPTION_LIMIT((L)); \
      const char *high_str = FORMAT_OPTION_LIMIT((H)); \
      sprintf(buffer, "--%s=%s..%s", #N, low_str, high_str); \
    } \
    int value = solver ? solver->options.N : (V); \
    const char *val_str = kissat_format_value(&format, b, value); \
    kissat_printf_usage(buffer, "%s [%s]", D, val_str); \
  } while (0);
  OPTIONS
#undef OPTION
}

bool kissat_options_parse_arg(const char *arg, char *buffer, int *val_ptr) {
  if (arg[0] != '-' || arg[1] != '-') {
    return false;
  }
  char const *name = arg + 2, *p = name;
  int ch;
  while ((ch = *p) && ch != '=') {
    p++;
  }
  if (ch) {
    assert(ch == '=');
    const size_t len = p - name;
    if (len >= kissat_options_max_name_buffer_size) {
      return false;
    }
    memcpy(buffer, name, len);
    buffer[len] = 0;
    const opt *const o = kissat_options_has(buffer);
    if (!o) {
      return false;
    }
    int value;
    if (!kissat_parse_option_value(p + 1, &value)) {
      return false;
    }
    if (value < o->low || value > o->high) {
      return false;
    }
    *val_ptr = value;
  } else {
    int value = 0;
    if (arg[2] == 'n' && arg[3] == 'o' && arg[4] == '-') {
      name += 3;
      const opt *const o = kissat_options_has(name);
      if (!o || o->low > (value = 0)) {
        return false;
      }
    } else {
      const opt *const o = kissat_options_has(name);
      if (!o || o->high < (value = 1)) {
        return false;
      }
    }
    assert(strlen(name) < kissat_options_max_name_buffer_size);
    strcpy(buffer, name);
    *val_ptr = value;
  }
  return true;
}

static bool ignore_embedded_option_for_fuzzing(const char *name) {
#ifdef EMBEDDED
  if (!strcmp(name, "embedded")) {
    return true;
  }
#endif
#ifndef QUIET
  if (!strcmp(name, "quiet")) {
    return true;
  }
#endif
  (void) name;
  return false;
}

void kissat_print_embedded_option_list(void) {
#define OPTION(N,V,L,H,D,F) \
  if (!ignore_embedded_option_for_fuzzing (#N)) \
    printf ("c --%s=%d\n", #N, (int) (V));
  OPTIONS
#undef OPTION
}

static bool ignore_range_option_for_fuzzing(const char *name) {
#ifdef LOGGING
  if (!strcmp(name, "log")) {
    return true;
  }
#endif
#ifdef EMBEDDED
  if (!strcmp(name, "embedded")) {
    return true;
  }
#endif
#ifndef QUIET
  if (!strcmp(name, "quiet")) {
    return true;
  }
#endif
  if (!strcmp(name, "reduce")) {
    return true;
  }
  if (!strcmp(name, "reluctant")) {
    return true;
  }
  if (!strcmp(name, "rephase")) {
    return true;
  }
  if (!strcmp(name, "restart")) {
    return true;
  }
  return false;
}

void kissat_print_option_range_list(void) {
#define OPTION(N,V,L,H,D,F) \
  if (!ignore_range_option_for_fuzzing (#N)) \
    printf ("%s %d %d %d\n", #N, (int)(L), (int) (V), (int)(H));
  OPTIONS
#undef OPTION
}

#endif
