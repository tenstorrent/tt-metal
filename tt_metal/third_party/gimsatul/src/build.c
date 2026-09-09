#include "build.h"
#include "config.h"
#include "message.h"

#include <stdio.h>

void print_banner (void) {
  if (verbosity < 0)
    return;
  printf ("c GimSATul SAT Solver\n");
  printf ("c Copyright (c) 2022-2023 Armin Biere University of Freiburg\n");
  fputs ("c\n", stdout);
  printf ("c Version %s%s%s\n", VERSION, GITID ? " " : "",
          GITID ? GITID : "");
  printf ("c %s\n", COMPILER);
  printf ("c %s\n", BUILD);
}

void print_version (void) { printf ("%s\n", VERSION); }

void print_id (void) { printf ("%s\n", GITID ? GITID : "unknown"); }
