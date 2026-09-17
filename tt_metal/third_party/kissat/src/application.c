#include "application.h"
#include "check.h"
#include "colors.h"
#include "config.h"
#include "error.h"
#include "internal.h"
#include "parse.h"
#include "print.h"
#include "proof.h"
#include "resources.h"
#include "witness.h"

#include <inttypes.h>
#include <string.h>
#include <unistd.h>

typedef struct application application;

struct application {
  kissat *solver;
  const char *input_path;
#ifndef NPROOFS
  const char *proof_path;
  file proof_file;
  int binary;
#endif
#if !defined(NPROOFS) || !defined (_POSIX_C_SOURCE)
  bool force;
#endif
  int time;
  int conflicts;
  int decisions;
  strictness strict;
  bool partial;
  bool witness;
  int max_var;
};

static void init_app(application *application, kissat *solver) {
  memset(application, 0, sizeof * application);
  application->solver = solver;
  application->witness = true;
  application->time = 0;
  application->conflicts = -1;
  application->decisions = -1;
  application->strict = NORMAL_PARSING;
}

static void print_common_dimacs_and_proof_usage(void) {
  printf("\n");
  printf("Furthermore '<dimacs>' is the input file in DIMACS format.\n");
#ifndef NPROOFS
  printf("If '<proof>' is specified then a proof trace is written.\n");
#endif
}

static void print_complete_dimacs_and_proof_usage(void) {
  printf("\n");
  printf("Furthermore '<dimacs>' is the input file in DIMACS format.\n");
#ifdef _POSIX_C_SOURCE
  printf("The solver reads from '<stdin>' if '<dimacs>' is unspecified.\n");
  printf("If the path has a '.bz2', '.gz', '.lzma', '7z' or '.xz' suffix\n");
  printf("then the solver tries to find a corresponding decompression\n");
  printf("tool ('bzip2', 'gzip', 'lzma', '7z', or 'xz') to decompress\n");
  printf("the input file on-the-fly after checking that the input file\n");
  printf("has the correct format (starts with the corresponding\n");
  printf("signature bytes).\n");
#endif
  printf("\n");
#ifndef NPROOFS
  printf("If '<proof>' is specified then a proof trace is written to the\n");
  printf("given file.  If the file name is '-' then the proof is written\n");
  printf
  ("to '<stdout>'. In this case the ASCII version of the DRAT format\n");
  printf
  ("is used.  For real files the binary proof format is used unless\n");
  printf("'--no-binary' is specified.\n");
  printf("\n");
#ifdef _POSIX_C_SOURCE
  printf("Writing of compressed proof files follows the same principle\n");
  printf("as reading compressed files. The compression format is based\n");
  printf("on the file suffix and it is checked that the corresponding\n");
  printf("compression utility can be found.\n");
#else
  printf("The solver was built without POSIX support. Thus compressed\n");
  printf("and unlocked reading and writing are not available. This is\n");
  printf("usually enforced by the '-p' (pedantic) configuration. If you\n");
  printf("need compressed reading and writing then configure and build\n");
  printf("the solver without '-p'. This will also speed-up file I/O.\n");
#endif
#else
  printf("The solver was built without proof support. If you need proofs\n");
  printf("use a configuration without '--no-proofs' nor '--ultimate'.\n");
#endif
}

static void print_force_usage(void) {
#if !defined(NPROOFS) && defined(_POSIX_C_SOURCE)
  printf("  -f      force writing proofs (to existing CNF alike file)\n");
#elif !defined(NPROOFS) && !defined(_POSIX_C_SOURCE)
  printf("  -f      force writing proofs or reading compressed files\n");
#elif defined(NPROOFS) && !defined(_POSIX_C_SOURCE)
  printf("  -f      force reading compressed as uncompressed files\n");
#endif
}

static void print_common_usage(void) {
  printf("usage: kissat [ <option> ... ] [ <dimacs> "
#ifndef NPROOFS
        "[ <proof> ] "
#endif
        "]\n"
        "\n"
        "where '<option>' is one of the following common options:\n"
        "\n"
        "  -h      print this list of common command line options\n"
        "  --help  print complete list of command line options\n");
  printf("\n");
  print_force_usage();
#if !defined(QUIET) && defined(LOGGING)
  printf("  -l      increase logging level (implies '-v' twice)\n");
#endif
  printf("  -n      do not print satisfying assignment\n");
#ifndef QUIET
  printf("\n");
  printf("  -q      suppress all messages\n");
  printf("  -s      print complete statistics\n");
  printf("  -v      increase verbose level\n");
#endif
  print_common_dimacs_and_proof_usage();
}

static void print_complete_usage(kissat *solver) {
  printf("usage: kissat [ <option> ... ] [ <dimacs> "
#ifndef NPROOFS
        "[ <proof> ] "
#endif
        "]\n"
        "\n"
        "where '<option>' is one of the following common options:\n"
        "\n"
        "  --help  print this list of all command line options\n"
        "  -h      print only reduced list of command line options\n");
  printf("\n");
  print_force_usage();
#if !defined(QUIET) && defined(LOGGING)
  printf("  -l      print logging messages"
#ifndef NOPTIONS
        " (see also '--log')"
#endif
        "\n");
#endif
  printf("  -n      do not print satisfying assignment\n");
#ifndef QUIET
  printf("\n");
  printf("  -q      suppress all messages"
#ifndef NOPTIONS
        " (see also '--quiet')"
#endif
        "\n");
  printf("  -s      print all statistics"
#ifndef NOPTIONS
        " (see also '--statistics')"
#endif
        "\n");
  printf("  -v      increase verbose level"
#ifndef NOPTIONS
        " (see also '--verbose')"
#endif
        "\n");
#endif
  printf("\n");
  printf("Further '<option>' can be one of the "
        "following less frequent options:\n");
  printf("\n");
  printf("  --banner             print solver information\n");
  printf("  --build              print build information\n");
  printf("  --color              "
        "use colors (default if connected to terminal)\n");
  printf("  --no-color           "
        "no colors (default if not connected to terminal)\n");
  printf("  --compiler           print compiler information\n");
  printf("  --copyright          print copyright information\n");
#if !defined(NOPTIONS) && defined(EMBEDDED)
  printf("  --embedded           print embedded option list\n");
#endif
#ifndef NPROOFS
  printf("  --force              same as '-f' (force writing proof)\n");
#endif
  printf("  --id                 print 'git' identifier (SHA-1 hash)\n");
#ifndef NOPTIONS
  printf("  --range              print option range list\n");
#endif
  printf("  --relaxed            relaxed parsing"
        " (ignore DIMACS header)\n");
  printf("  --strict             stricter parsing"
        " (no empty header lines)\n");
  printf("  --version            print version\n");
  printf("\n");
  printf("The following solving limits can be enforced:\n");
  printf("\n");
  printf("  --conflicts=<limit>\n");
  printf("  --decisions=<limit>\n");
  printf("  --time=<seconds>\n");
  printf("\n");
  printf
  ("Satisfying assignments have by default values for all variables\n");
  printf("unless '--partial' is specified, then only values are printed\n");
  printf("for variables which are necessary to satisfy the formula.\n");
  printf("\n");
#ifndef NOPTIONS
  printf
  ("The following predefined 'configurations' (option settings) are supported:\n");
  printf("\n");
  kissat_configuration_usage();
  printf("\n");
  printf("Or '<option>' is one of the following long options:\n\n");
  kissat_options_usage(solver);
#else
  printf("The solver was configured without options ('--no-options').\n");
  printf("Thus all internal options are fixed and can not be changed.\n");
  printf("If you want to change them at run-time use a configuration\n");
  printf("without '--no-options'. Note, that '--extreme', '-competition'\n");
  printf("as well as '--ultimate' all enforce '--no-options' as well.\n");
#ifdef SAT
  printf("The '--sat' option is ignored since set at compile time.\n");
#elif UNSAT
  printf("The '--unsat' option is ignored since set at compile time.\n");
#else
  printf("The '--default' option is ignored but allowed.\n");
#endif
#endif
  print_complete_dimacs_and_proof_usage();
}

static bool parsed_one_option_and_return_zero_exit_code(
      kissat *solver, char *arg) {
  if (!strcmp(arg, "-h")) {
    print_common_usage();
    return true;
  }
  if (!strcmp(arg, "--help")) {
    print_complete_usage(solver);
    return true;
  }
  if (!strcmp(arg, "--banner")) {
    kissat_banner(0, "Kissat Extras SAT Solver");
    return true;
  }
  if (!strcmp(arg, "--build")) {
    kissat_build(0);
    return true;
  }
  if (!strcmp(arg, "--copyright")) {
    printf("%s\n", kissat_copyright());
    return true;
  }
  if (!strcmp(arg, "--compiler")) {
    printf("%s\n", kissat_compiler());
    return true;
  }
#if !defined(NOPTIONS) && defined(EMBEDDED)
  if (!strcmp(arg, "--embedded")) {
    kissat_print_embedded_option_list();
    return true;
  }
#endif
  if (!strcmp(arg, "--id")) {
    printf("%s\n", kissat_id());
    return true;
  }
#ifndef NOPTIONS
  if (!strcmp(arg, "--range")) {
    kissat_print_option_range_list();
    return true;
  }
#endif
  if (!strcmp(arg, "--version")) {
    printf("%s\n", kissat_version());
    return true;
  }
  return false;
}

static const char *single_first_option_table[] = {
  "-h",
  "--help",
  "--banner",
  "--build",
  "--copyright",
  "--compiler",
#if !defined(NOPTIONS) && defined(EMBEDDED)
  "--embedded",
#endif
  "--id",
#ifndef NOPTIONS
  "--range",
#endif
  "--version"
};

static bool single_first_option(const char *arg) {
  const unsigned size = sizeof single_first_option_table / sizeof(char *);
  for (unsigned i = 0; i < size; i++)
    if (!strcmp(single_first_option_table[i], arg)) {
      return true;
    }
  return false;
}

#define ERROR(...) \
do { \
  kissat_error (__VA_ARGS__); \
  return false; \
} while (0)

#ifndef NPROOFS

static bool most_likely_existing_cnf_file(const char *path) {
  if (!kissat_file_readable(path)) {
    return false;
  }

  if (kissat_has_suffix(path, ".dimacs")) {
    return true;
  }
  if (kissat_has_suffix(path, ".dimacs.7z")) {
    return true;
  }
  if (kissat_has_suffix(path, ".dimacs.bz2")) {
    return true;
  }
  if (kissat_has_suffix(path, ".dimacs.gz")) {
    return true;
  }
  if (kissat_has_suffix(path, ".dimacs.lzma")) {
    return true;
  }
  if (kissat_has_suffix(path, ".dimacs.xz")) {
    return true;
  }

  if (kissat_has_suffix(path, ".cnf")) {
    return true;
  }
  if (kissat_has_suffix(path, ".cnf.7z")) {
    return true;
  }
  if (kissat_has_suffix(path, ".cnf.bz2")) {
    return true;
  }
  if (kissat_has_suffix(path, ".cnf.gz")) {
    return true;
  }
  if (kissat_has_suffix(path, ".cnf.lzma")) {
    return true;
  }
  if (kissat_has_suffix(path, ".cnf.xz")) {
    return true;
  }

  return false;
}

#endif

#ifndef NPROOFS

#define LONG_FALSE_OPTION(ARG,NAME) \
  (!strcmp ((ARG), "--no-" NAME) || \
   !strcmp ((ARG), "--" NAME "=0") || \
   !strcmp ((ARG), "--" NAME "=false"))

#endif

#define LONG_TRUE_OPTION(ARG,NAME) \
  (!strcmp ((ARG), "--" NAME) || \
   !strcmp ((ARG), "--" NAME "=1") || \
   !strcmp ((ARG), "--" NAME "=true"))

static bool parse_options(application *application, int argc, char **argv) {
  kissat *solver = application->solver;
  const char *strict_option = 0;
#ifndef NOPTIONS
  const char *configuration = 0;
#endif
#if !defined (NPROOFS) || !defined (_POSIX_C_SOURCE)
  const char *force_option = 0;
#endif
  const char *valstr;
  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (single_first_option(arg))
      ERROR("option '%s' only allowed as %s argument",
            arg, i == 1 ? "single" : "first");
#if !defined (NPROOFS) || !defined (_POSIX_C_SOURCE)
    else if (!strcmp(arg, "-f") ||
          LONG_TRUE_OPTION(arg, "force") ||
          LONG_TRUE_OPTION(arg, "forced")) {
      if (application->force) {
        assert(force_option);
        if (!strcmp(force_option, arg)) {
          ERROR("multiple '%s' options", force_option);
        } else
          ERROR("'%s' and '%s' have the same effect",
                force_option, arg);
      }
      application->force = true;
      force_option = arg;
    }
#endif
    else if (LONG_TRUE_OPTION(arg, "relax") ||
          LONG_TRUE_OPTION(arg, "relaxed")) {
      if (strict_option) {
        if (application->strict != RELAXED_PARSING)
          ERROR("can not combine contradictory '%s' and '%s'",
                strict_option, arg);
        else if (!strcmp(strict_option, arg)) {
          ERROR("multiple '%s' options", strict_option);
        } else
          ERROR("'%s' and '%s' have the same effect",
                strict_option, arg);
      }
      application->strict = RELAXED_PARSING;
      strict_option = arg;
    } else if (LONG_TRUE_OPTION(arg, "strict") ||
          LONG_TRUE_OPTION(arg, "stricter") ||
          LONG_TRUE_OPTION(arg, "pedantic")) {
      if (strict_option) {
        if (application->strict != PEDANTIC_PARSING)
          ERROR("can not combine contradictory '%s' and '%s'",
                strict_option, arg);
        else if (!strcmp(strict_option, arg)) {
          ERROR("multiple '%s' options", strict_option);
        } else
          ERROR("'%s' and '%s' have the same effect",
                strict_option, arg);
      }
      application->strict = PEDANTIC_PARSING;
      strict_option = arg;
    }
#if defined(LOGGING) && !defined(QUIET) && !defined(NOPTIONS)
    else if (!strcmp(arg, "-l")) {
      int value = GET_OPTION(log);
      if (value < INT_MAX) {
        value++;
      }
      kissat_set_option(solver, "log", value);
    }
#endif
    else if (!strcmp(arg, "-n")) {
      application->witness = false;
    }
#if !defined(QUIET) && !defined(NOPTIONS)
    else if (!strcmp(arg, "-q")) {
      kissat_set_option(solver, "quiet", 1);
    } else if (!strcmp(arg, "-s")) {
      kissat_set_option(solver, "statistics", 1);
    } else if (!strcmp(arg, "-v")) {
      int value = GET_OPTION(verbose);
      if (value < INT_MAX) {
        value++;
      }
      kissat_set_option(solver, "verbose", value);
    }
#endif
    else if (!strcmp(arg, "--color") ||
          !strcmp(arg, "--colors") ||
          !strcmp(arg, "--colour") || !strcmp(arg, "--colours")) {
      kissat_force_colors();
    } else if (!strcmp(arg, "--no-color") ||
          !strcmp(arg, "--no-colors") ||
          !strcmp(arg, "--no-colour") || !strcmp(arg, "--no-colours")) {
      kissat_force_no_colors();
    } else if ((valstr = kissat_parse_option_name(arg, "time"))) {
      int val;
      if (kissat_parse_option_value(valstr, &val) && val > 0) {
        if (application->time > 0)
          ERROR("multiple '--time=%d' and '%s'",
                application->time, arg);
        application->time = val;
        alarm(val);
      } else {
        ERROR("invalid argument in '%s' (try '-h')", arg);
      }
    } else if ((valstr = kissat_parse_option_name(arg, "conflicts"))) {
      int val;
      if (kissat_parse_option_value(valstr, &val) && val >= 0) {
        if (application->conflicts >= 0)
          ERROR("multiple '--conflicts=%d' and '%s'",
                application->conflicts, arg);
        kissat_set_conflict_limit(solver, val);
        application->conflicts = val;
      } else {
        ERROR("invalid argument in '%s' (try '-h')", arg);
      }
    } else if ((valstr = kissat_parse_option_name(arg, "decisions"))) {
      int val;
      if (kissat_parse_option_value(valstr, &val) && val >= 0) {
        if (application->decisions >= 0)
          ERROR("multiple '--decisions=%d' and '%s'",
                application->decisions, arg);
        kissat_set_decision_limit(solver, val);
        application->decisions = val;
      } else {
        ERROR("invalid argument in '%s' (try '-h')", arg);
      }
    } else if (!strcmp(arg, "--partial")) {
      application->partial = true;
    }
#ifndef NPROOFS
    else if (LONG_FALSE_OPTION(arg, "binary")) {
      application->binary = -1;
    }
#endif
#ifndef NOPTIONS
    else if (arg[0] == '-' && arg[1] == '-' &&
          kissat_has_configuration(arg + 2)) {
      if (configuration)
        ERROR("multiple configurations '%s' and '%s'",
              configuration, arg);
      kissat_set_configuration(solver, arg + 2);
      configuration = arg;
    } else if (arg[0] == '-' && arg[1] == '-') {
      char name[kissat_options_max_name_buffer_size];
      int value;
      if (!kissat_options_parse_arg(arg, name, &value)) {
        ERROR("invalid long option '%s' (try '-h')", arg);
      }
      kissat_set_option(solver, name, value);
    }
#else
#ifdef SAT
    else if (!strcmp(arg, "--sat"))
      ;
#elif defined(UNSAT)
    else if (!strcmp(arg, "--unsat"))
      ;
#else
    else if (!strcmp(arg, "--default"))
      ;
#endif
    else if (arg[0] == '-' && arg[1] == '-')
      ERROR("invalid long option '%s' "
            "(configured with '--no-options')", arg);
#endif
#ifdef QUIET
    else if (arg[0] == '-' && !arg[2] &&
          (arg[1] == 'q' || arg[1] == 's' || arg[1] == 'v')) {
      ERROR("invalid short option '%s' (configured with '-q')", arg);
    }
#endif
#ifndef LOGGING
    else if (!strcmp(arg, "-l"))
      ERROR("invalid short option '%s' "
            "(configured without '-l' or '-g')", arg);
#endif
#ifdef NOPTIONS
    else if (arg[0] == '-' && !arg[2] &&
          (arg[1] == 'l' || arg[1] == 'q' ||
                arg[1] == 's' || arg[1] == 'v'))
      ERROR("invalid short option '%s' "
            "(configured with '--no-options')", arg);
#endif
    else if (arg[0] == '-' && arg[1]) {
      ERROR("invalid short option '%s' (try '-h')", arg);
    }
#ifndef NPROOFS
    else if (application->proof_path)
      ERROR("three file arguments '%s', '%s' and '%s' (try '-h')",
            application->input_path, application->proof_path, arg);
#endif
    else if (application->input_path) {
#ifndef NPROOFS
      const char *input_path = application->input_path;
      if (!strcmp(input_path, arg))
        ERROR("will not read and write '%s' at the same time",
              input_path);
#ifdef _POSIX_C_SOURCE
      {
        char *real_input_path = realpath(input_path, 0);
        if (real_input_path) {
          char *real_arg_path = realpath(arg, 0);
          if (real_arg_path) {
            if (!strcmp(real_input_path, real_arg_path)) {
              if (strcmp(arg, real_arg_path) &&
                    strcmp(input_path, real_input_path))
                ERROR("will not read and write '%s' and '%s' "
                      "pointing to the same file '%s'",
                      input_path, arg, real_input_path);
              else
                ERROR("will not read and write '%s' and '%s' "
                      "pointing to the same file",
                      input_path, arg);
            }
            free(real_arg_path);
          }
          free(real_input_path);
        } else
          ERROR("can not get absolute path of '%s' (unexpectedly)",
                input_path);
      }
#endif
      if (!application->force && most_likely_existing_cnf_file(arg)) {
        ERROR("not writing proof to '%s' file (use '-f')", arg);
      }
      if (!kissat_file_writable(arg)) {
        ERROR("can not write proof to '%s'", arg);
      }
      application->proof_path = arg;
#else
      ERROR("two file arguments '%s' and '%s' without proof support "
            "(try '-h')", application->input_path, arg);
#endif
    } else {
      if (!kissat_file_readable(arg)) {
        ERROR("can not read '%s'", arg);
      }
      application->input_path = arg;
    }
  }
#ifndef _POSIX_C_SOURCE
  if (!application->force &&
        application->input_path &&
        kissat_looks_like_a_compressed_file(application->input_path))
    ERROR("reading apparently compressed '%s' not supported "
          "(use '-f' to force reading without decompression)",
          application->input_path);
#endif
#if !defined(QUIET) && !defined(NOPTIONS)
  if (kissat_get_option(solver, "quiet")) {
    if (kissat_get_option(solver, "statistics")) {
      ERROR("can not use '--quiet' ('-q') with '--statistics' ('-s')");
    }
    if (kissat_get_option(solver, "verbose")) {
      ERROR("can not use '--quiet' ('-q') with '--verbose' ('-v')");
    }
  }
#endif
  return true;
}

static bool parse_input(application *application) {
#ifndef QUIET
  double entered = kissat_process_time();
#endif
  kissat *solver = application->solver;
  uint64_t lineno;
  file file;
  const char *path = application->input_path;
  if (!path) {
    kissat_read_already_open_file(&file, stdin, "<stdin>");
  } else if (!kissat_open_to_read_file(&file, path)) {
    ERROR("failed to open '%s' for reading", path);
  }
  kissat_section(solver, "parsing");
  kissat_message(solver, "opened and reading %sDIMACS file:",
        file.compressed ? "compressed " : "");
  kissat_line(solver);
  kissat_message(solver, "  %s", file.path);
  kissat_line(solver);
  const char *error = kissat_parse_dimacs(solver, application->strict, &file,
              &lineno, &application->max_var);
  kissat_close_file(&file);
  if (error) {
    ERROR("%s:%" PRIu64 ": parse error: %s", file.path, lineno, error);
  }
#ifndef QUIET
  kissat_message(solver, "closing input after reading %s",
        FORMAT_BYTES(file.bytes));
  if (file.compressed) {
    assert(path);
    size_t bytes = kissat_file_size(path);
    kissat_message(solver,
          "inflated input file of size %s by %.2f",
          FORMAT_BYTES(bytes),
          kissat_average(file.bytes, bytes));
  }
  kissat_message(solver,
        "finished parsing after %.2f seconds",
        kissat_process_time() - entered);
#endif
  return true;
}

#ifndef NPROOFS

static bool write_proof(application *application) {
  const char *path = application->proof_path;
  if (!path) {
    return true;
  }
  file *file = &application->proof_file;
  bool binary = true;
  if (!strcmp(path, "-")) {
    binary = false;
    kissat_write_already_open_file(file, stdout, "<stdout>");
  } else if (!kissat_open_to_write_file(file, path)) {
    ERROR("failed to open and write proof to '%s'", path);
  } else if (application->binary < 0) {
    binary = false;
  }
  kissat_init_proof(application->solver, file, binary);
#ifndef QUIET
  kissat *solver = application->solver;
  kissat_section(solver, "proving");
  kissat_message(solver, "%swriting proof to %sDRAT file:",
        file->close ? "opened and " : "",
        file->compressed ? "compressed " : "");
  kissat_line(solver);
  kissat_message(solver, "  %s", file->path);
#endif
  return true;
}

static void close_proof(application *application) {
  const char *path = application->proof_path;
  if (!path) {
    return;
  }
  kissat_release_proof(application->solver);
  kissat_close_file(&application->proof_file);
}

#endif

#ifndef QUIET

#ifndef NOPTIONS
static void print_option(kissat *solver, int value, const opt *o) {
  char buffer[96];
  const bool b = (o->low == 0 && o->high == 1);
  const char *val_str = FORMAT_VALUE(b, value);
  const char *def_str = FORMAT_VALUE(b, o->value);
  sprintf(buffer, "%s=%s", o->name, val_str);
  kissat_message(solver, "--%-30s (%s default '%s')",
        buffer, (value == o->value ? "same as" : "different from"),
        def_str);
}
#endif

#ifndef NOPTIONS
static void print_options(kissat *solver) {
  const int verbosity = kissat_verbosity(solver);
  if (verbosity < 0) {
    return;
  }
  size_t printed = 0;
  for (all_options(o)) {
    const int value = *kissat_options_ref(&solver->options, o);
    if (o->value != value || verbosity > 0) {
      if (!printed++) {
        kissat_section(solver, "options");
      }

      print_option(solver, value, o);
    }
  }
}
#endif

static void print_limits(application *application) {
  kissat *solver = application->solver;
  const int verbosity = kissat_verbosity(solver);
  if (verbosity < 1 &&
        application->conflicts < 0 && application->decisions < 0) {
    return;
  }

  kissat_section(solver, "limits");
  if (!application->time &&
        application->conflicts < 0 && application->decisions < 0) {
    kissat_message(solver, "no time, conflict nor decision limit set");
  } else {
    if (application->time)
      kissat_message(solver,
            "time limit set to %d seconds", application->time);
    else if (verbosity > 0) {
      kissat_message(solver, "no time limit");
    }

    if (application->conflicts >= 0)
      kissat_message(solver,
            "conflict limit set to %d conflicts",
            application->conflicts);
    else if (verbosity > 0) {
      kissat_message(solver, "no conflict limit");
    }

    if (application->decisions >= 0)
      kissat_message(solver,
            "decision limit set to %d decisions",
            application->decisions);
    else if (verbosity > 0) {
      kissat_message(solver, "no decision limit");
    }
  }
}

#endif

static int run_application(kissat *solver,
      int argc, char **argv, bool *cancel_alarm_ptr) {
  *cancel_alarm_ptr = false;
  if (argc == 2)
    if (parsed_one_option_and_return_zero_exit_code(solver, argv[1])) {
      return 0;
    }
  application application;
  init_app(&application, solver);
  bool ok = parse_options(&application, argc, argv);
  if (application.time > 0) {
    *cancel_alarm_ptr = true;
  }
  if (!ok) {
    return 1;
  }
#ifndef QUIET
  kissat_section(solver, "banner");
  if (!GET_OPTION(quiet)) {
    kissat_banner("c ", "Kissat Extras SAT Solver");
    fflush(stdout);
  }
#endif
#ifndef NPROOFS
  if (!write_proof(&application)) {
    return 1;
  }
#endif
  if (!parse_input(&application)) {
#ifndef NPROOFS
    close_proof(&application);
#endif
    return 1;
  }
#ifndef QUIET
#ifndef NOPTIONS
  print_options(solver);
#endif
  print_limits(&application);
  kissat_section(solver, "solving");
#endif
  int res = kissat_solve(solver);
  if (res) {
    kissat_section(solver, "result");
    if (res == 20) {
      printf("s UNSATISFIABLE\n");
      fflush(stdout);
    } else if (res == 10) {
#ifndef NDEBUG
      if (GET_OPTION(check)) {
        kissat_check_satisfying_assignment(solver);
      }
#endif
      printf("s SATISFIABLE\n");
      fflush(stdout);
      if (application.witness)
        kissat_print_witness(solver,
              application.max_var, application.partial);
    }
  }
#ifndef QUIET
  kissat_print_statistics(solver);
#endif
#ifndef NPROOFS
  close_proof(&application);
#endif
#ifndef QUIET
  kissat_section(solver, "shutting down");
  kissat_message(solver, "exit %d", res);
#endif
  return res;
}

int kissat_application(kissat *solver, int argc, char **argv) {
  bool cancel_alarm;
  int res = run_application(solver, argc, argv, &cancel_alarm);
  if (cancel_alarm) {
    alarm(0);
  }
  return res;
}
