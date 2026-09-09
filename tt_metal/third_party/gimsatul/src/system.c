#include "system.h"
#include "utilities.h"

#include <assert.h>
#include <stdio.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

double process_time (void) {
  struct rusage u;
  double res;
  if (getrusage (RUSAGE_SELF, &u))
    return 0;
  res = u.ru_utime.tv_sec + 1e-6 * u.ru_utime.tv_usec;
  res += u.ru_stime.tv_sec + 1e-6 * u.ru_stime.tv_usec;
  return res;
}

double current_time (void) {
  struct timeval tv;
  if (gettimeofday (&tv, 0))
    return 0;
  return 1e-6 * tv.tv_usec + tv.tv_sec;
}

double start_time;

double wall_clock_time (void) { return current_time () - start_time; }

size_t maximum_resident_set_size (void) {
  struct rusage u;
  if (getrusage (RUSAGE_SELF, &u))
    return 0;
  size_t res = (size_t) u.ru_maxrss;
#ifdef __APPLE__
  return res;
#else
  return res << 10;
#endif
}

#ifdef __APPLE__

#include <mach/task.h>

mach_port_t mach_task_self (void);

size_t current_resident_set_size (void) {
  struct task_basic_info info;
  mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
  if (KERN_SUCCESS != task_info (mach_task_self (), TASK_BASIC_INFO,
                                 (task_info_t) &info, &count))
    return 0;
  return info.resident_size;
}

#else

size_t current_resident_set_size (void) {
  char path[48];
  sprintf (path, "/proc/%d/statm", (int) getpid ());
  FILE *file = fopen (path, "r");
  if (!file)
    return 0;
  size_t dummy, rss;
  int scanned = fscanf (file, "%zu %zu", &dummy, &rss);
  fclose (file);
  return scanned == 2 ? rss * sysconf (_SC_PAGESIZE) : 0;
}

#endif

void summarize_used_resources (unsigned t) {
  assert (t);
  double w = current_time () - start_time;
  double p = process_time ();
  double m = maximum_resident_set_size () / (double) (1u << 20);
  double u = percent (p, w) / t;
  printf ("c resources: %.0f%% utilization = %.2f / %.2f sec / %u threads, "
          "%.2f MB\n",
          u, p, w, t, m);
  fflush (stdout);
}
