#include "resources.h"

#include <sys/time.h>

double kissat_wall_clock_time(void) {
  struct timeval tv;
  if (gettimeofday(&tv, 0)) {
    return 0;
  }
  return 1e-6 * tv.tv_usec + tv.tv_sec;
}

#ifndef QUIET

#include "internal.h"
#include "statistics.h"
#include "utilities.h"

#include <stdio.h>
#include <inttypes.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

double kissat_process_time(void) {
  struct rusage u;
  double res;
  if (getrusage(RUSAGE_SELF, &u)) {
    return 0;
  }
  res = u.ru_utime.tv_sec + 1e-6 * u.ru_utime.tv_usec;
  res += u.ru_stime.tv_sec + 1e-6 * u.ru_stime.tv_usec;
  return res;
}

uint64_t kissat_maximum_resident_set_size(void) {
  struct rusage u;
  if (getrusage(RUSAGE_SELF, &u)) {
    return 0;
  }
  return ((uint64_t) u.ru_maxrss) << 10;
}

uint64_t kissat_current_resident_set_size(void) {
  char path[48];
  sprintf(path, "/proc/%" PRIu64 "/statm", (uint64_t) getpid());
  FILE *file = fopen(path, "r");
  if (!file) {
    return 0;
  }
  uint64_t dummy, rss;
  int scanned = fscanf(file, "%" PRIu64 " %" PRIu64 "", &dummy, &rss);
  fclose(file);
  return scanned == 2 ? rss * sysconf(_SC_PAGESIZE) : 0;
}

void kissat_print_resources(kissat *solver) {
  uint64_t rss = kissat_maximum_resident_set_size();
  double t = kissat_time(solver);
  printf("c "
        "%-" SFW1 "s "
        "%" SFW2 PRIu64 " "
        "%-" SFW3 "s "
        "%" SFW4 ".0f "
        "MB\n",
        "maximum-resident-set-size:",
        rss, "bytes", rss / (double)(1 << 20));
#ifdef METRICS
  statistics *statistics = &solver->statistics;
  uint64_t max_allocated = statistics->allocated_max + sizeof(kissat);
  printf("c "
        "%-" SFW1 "s "
        "%" SFW2 PRIu64 " "
        "%-" SFW3 "s "
        "%" SFW4 ".0f "
        "%%\n",
        "max-allocated:",
        max_allocated, "bytes", kissat_percent(max_allocated, rss));
#endif
  printf("c process-time: %30s %18.2f seconds\n", FORMAT_TIME(t), t);
  fflush(stdout);
}

#endif
