#include "build.h"
#include "colors.h"
#include "print.h"
#include "kissat.h"

#include <stdio.h>
#include <string.h>

const char *kissat_signature(void) {
  return "kissat-extras-" VERSION;
}

const char *kissat_id(void) {
  return ID;
}

const char *kissat_compiler(void) {
  return COMPILER;
}

const char *kissat_copyright(void) {
  return "Kissat SAT Solver: Copyright (c) 2019-2021 Armin Biere JKU Linz; "
        "Kissat Extras Patches: Copyright (c) 2022 Jannis Harder";
}

const char *kissat_version(void) {
  return VERSION;
}

#define PREFIX(COLORS) \
do { \
  if (prefix) \
    fputs (prefix, stdout); \
  COLOR (COLORS); \
} while (0)

#define NL() \
do { \
  COLOR (NORMAL); \
  fputs ("\n", stdout); \
  fflush(stdout); \
} while (0)

void kissat_build(const char *prefix) {
  TERMINAL(stdout, 1);
  if (!prefix) {
    connected_to_terminal = false;
  }

  PREFIX(MAGENTA);
  if (ID) {
    printf("Version %s %s", VERSION, ID);
  } else {
    printf("Version %s", VERSION);
  }
  NL();

  PREFIX(MAGENTA);
  printf("%s", COMPILER);
  NL();

  PREFIX(MAGENTA);
  printf("%s", BUILD);
  NL();
}

void kissat_banner(const char *prefix, const char *name) {
  TERMINAL(stdout, 1);
  if (!prefix) {
    connected_to_terminal = false;
  }

  PREFIX(BOLD MAGENTA);
  printf("%s", name);
  NL();

  const char *copyright = kissat_copyright();

  for (const char *end; end = strstr(copyright, "; "), end;) {
    PREFIX(BOLD MAGENTA);
    fwrite(copyright, end - copyright, 1, stdout);
    NL();
    copyright = end + 2;
  }

  PREFIX(BOLD MAGENTA);
  fputs(copyright, stdout);
  NL();

  if (prefix) {
    PREFIX("");
    NL();
  }

  kissat_build(prefix);
}
