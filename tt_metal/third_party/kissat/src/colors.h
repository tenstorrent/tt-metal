#ifndef _colors_h_INCLUDED
#define _colors_h_INCLUDED

#include <assert.h>
#include <stdio.h>
#include <stdbool.h>

#define BLUE "\033[34m"
#define BOLD "\033[1m"
#define CYAN "\033[36m"
#define GREEN "\033[32m"
#define MAGENTA "\033[35m"
#define NORMAL "\033[0m"
#define RED "\033[31m"
#define WHITE "\037[34m"
#define YELLOW "\033[33m"

#define LIGHT_GRAY "\033[0;37m"
#define DARK_GRAY "\033[1;30m"

#ifdef _POSIX_C_SOURCE
#define assert_if_posix assert
#else
#define assert_if_posix(...) do { } while (0)
#endif

#define TERMINAL(F,I) \
assert_if_posix (fileno (F) == I); /* 'fileno' only in POSIX not C99 */ \
assert ((I == 1 && F == stdout) || (I == 2 && F == stderr)); \
bool connected_to_terminal = kissat_connected_to_terminal (I); \
FILE * terminal_file = F

#define COLOR(CODE) \
do { \
  if (!connected_to_terminal) \
    break; \
  fputs (CODE, terminal_file); \
} while (0)

extern int kissat_is_terminal[3];

int kissat_initialize_terminal(int fd);
void kissat_force_colors(void);
void kissat_force_no_colors(void);

static inline bool kissat_connected_to_terminal(int fd) {
  assert(fd == 1 || fd == 2);
  int res = kissat_is_terminal[fd];
  if (res < 0) {
    res = kissat_initialize_terminal(fd);
  }
  assert(res == 0 || res == 1);
  return res;
}

static inline const char *kissat_bold_green_color_code(int fd) {
  return kissat_connected_to_terminal(fd) ? BOLD GREEN : "";
}

static inline const char *kissat_normal_color_code(int fd) {
  return kissat_connected_to_terminal(fd) ? NORMAL : "";
}

#endif
