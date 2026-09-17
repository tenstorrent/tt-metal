#include "format.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

char *kissat_next_format_string(format *format) {
  assert(format->pos < NUM_FORMAT_STRINGS);
  char *res = format->str[format->pos++];
  if (format->pos == NUM_FORMAT_STRINGS) {
    format->pos = 0;
  }
  return res;
}

static void format_count(char *res, uint64_t w) {
  if (w >= 128 && kissat_is_power_of_two(w)) {
    unsigned l;
    for (l = 0; ((uint64_t) 1 << l) != w; l++) {
      assert(l + 1 < 8 * sizeof(word));
    }
    sprintf(res, "2^%u", l);
  } else if (w >= 1000 && !(w % 1000)) {
    unsigned l;
    for (l = 0; !(w % 10); l++) {
      w /= 10;
    }
    sprintf(res, "%" PRIu64 "e%u", w, l);
  } else {
    sprintf(res, "%" PRIu64, w);
  }
}

const char *kissat_format_count(format *format, uint64_t w) {
  char *res = kissat_next_format_string(format);
  format_count(res, w);
  return res;
}

const char *kissat_format_value(format *format, bool boolean, int value) {
  if (boolean && value) {
    return "true";
  }
  if (boolean && !value) {
    return "false";
  }
  if (value == INT_MAX) {
    return "INT_MAX";
  }
  if (value == INT_MIN) {
    return "INT_MIN";
  }
  char *res = kissat_next_format_string(format);
  if (value < 0) {
    *res = '-';
    format_count(res + 1, ABS(value));
  } else {
    format_count(res, value);
  }
  return res;
}

const char *kissat_format_bytes(format *format, uint64_t bytes) {
  char *res = kissat_next_format_string(format);
  if (bytes < (1u << 10)) {
    sprintf(res, "%" PRIu64 " bytes", bytes);
  } else if (bytes < (1u << 20))
    sprintf(res, "%" PRIu64 " bytes (%" PRIu64 " KB)",
          bytes, (bytes + (1 << 9)) >> 10);
  else if (bytes < (1u << 30))
    sprintf(res, "%" PRIu64 " bytes (%" PRIu64 " MB)",
          bytes, (bytes + (1u << 19)) >> 20);
  else
    sprintf(res, "%" PRIu64 " bytes (%" PRIu64 " GB)",
          bytes, (bytes + (1u << 29)) >> 30);
  return res;
}

const char *kissat_format_time(format *format, double seconds) {
  if (!seconds) {
    return "0s";
  }
  char *res = kissat_next_format_string(format);
  uint64_t rounded = round(seconds);
  uint64_t minutes = rounded / 60;
  rounded %= 60;
  uint64_t hours = minutes / 60;
  minutes %= 60;
  uint64_t days = hours / 24;
  hours %= 24;
  char *tmp = res;
  if (days) {
    sprintf(res, "%" PRIu64 "d", days);
    tmp += strlen(res);
  }
  if (hours) {
    if (tmp != res) {
      *tmp++ = ' ';
    }
    sprintf(tmp, "%" PRIu64 "h", hours);
    tmp += strlen(tmp);
  }
  if (minutes) {
    if (tmp != res) {
      *tmp++ = ' ';
    }
    sprintf(tmp, "%" PRIu64 "m", minutes);
    tmp += strlen(tmp);
  }
  if (rounded) {
    if (tmp != res) {
      *tmp++ = ' ';
    }
    sprintf(tmp, "%" PRIu64 "s", rounded);
  }
  return res;
}

const char *kissat_format_signs(format *format, unsigned size, word signs) {
  char *res = kissat_next_format_string(format);
  assert(size + 1 < FORMAT_STRING_SIZE);
  char *p = res;
  word bit = 1;
  for (unsigned i = 0; i < size; i++, bit <<= 1) {
    *p++ = (bit & signs) ? '1' : '0';
  }
  *p = 0;
  return res;
}

const char *kissat_format_ordinal(format *format, uint64_t ordinal) {
  char const *suffix;
  unsigned mod100 = ordinal % 100;
  if (10 <= mod100 && mod100 <= 19) {
    suffix = "th";
  } else {
    switch (mod100 % 10) {
    case 1:
      suffix = "st";
      break;
    case 2:
      suffix = "nd";
      break;
    case 3:
      suffix = "rd";
      break;
    default:
      suffix = "th";
      break;
    }
  }
  char *res = kissat_next_format_string(format);
  sprintf(res, "%" PRIu64 "%s", ordinal, suffix);
  return res;
}
