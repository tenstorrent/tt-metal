#include "build.h"
#include "catch.h"
#include "clone.h"
#include "detach.h"
#include "message.h"
#include "parse.h"
#include "ruler.h"
#include "simplify.h"
#include "solve.h"
#include "statistics.h"
#include "system.h"
#include "types.h"
#include "witness.h"

#include "cover.h"

int main (int argc, char **argv) {
  start_time = current_time ();
  struct options options;
  parse_options (argc, argv, &options);
  print_banner ();
  check_types ();
  if (verbosity >= 0 && options.proof.file) {
    printf ("c\nc writing %s proof trace to '%s'\n",
            options.binary ? "binary" : "ASCII", options.proof.path);
    fflush (stdout);
  }
  int variables, clauses;
  parse_dimacs_header (&options, &variables, &clauses);
  struct ruler *ruler = new_ruler (variables, &options);
  set_signal_handlers (ruler);
  parse_dimacs_body (ruler, variables, clauses);
  report_non_default_options (&options);
  simplify_ruler (ruler);
  clone_rings (ruler);
  struct ring *winner = solve_rings (ruler);
  int res = winner ? winner->status : 0;
  reset_signal_handlers ();
  close_proof (&options.proof);
  if (res == 20) {
    if (verbosity >= 0)
      printf ("c\n");
    printf ("s UNSATISFIABLE\n");
    fflush (stdout);
  } else if (res == 10) {
    signed char *witness = extend_witness (winner);
    check_witness (witness, ruler->original);
    if (verbosity >= 0)
      printf ("c\n");
    printf ("s SATISFIABLE\n");
    if (options.witness)
      print_witness (ruler->size, witness);
    fflush (stdout);
    free (witness);
  }
  if (options.summarize)
    summarize_used_resources (options.threads);
#ifndef QUIET
  else
    print_ruler_statistics (ruler);
#endif
  detach_and_delete_rings (ruler);
  delete_ruler (ruler);
  if (verbosity >= 0) {
    printf ("c\nc exit %d\n", res);
    fflush (stdout);
  }
  return res;
}
