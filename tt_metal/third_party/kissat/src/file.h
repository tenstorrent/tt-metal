#ifndef _file_h_INCLUDED
#define _file_h_INCLUDED

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

bool kissat_file_exists(const char *path);
bool kissat_file_readable(const char *path);
bool kissat_file_writable(const char *path);
size_t kissat_file_size(const char *path);
bool kissat_find_executable(const char *name);

typedef struct file file;

struct file {
  FILE *file;
  bool close;
  bool reading;
  bool compressed;
  const char *path;
  uint64_t bytes;
};

void kissat_read_already_open_file(file *, FILE *, const char *path);
void kissat_write_already_open_file(file *, FILE *, const char *path);

bool kissat_open_to_read_file(file *, const char *path);
bool kissat_open_to_write_file(file *, const char *path);

void kissat_close_file(file *);

#ifndef _POSIX_C_SOURCE

bool kissat_looks_like_a_compressed_file(const char *path);

#endif

static inline int kissat_getc(file *file) {
  assert(file);
  assert(file->file);
  assert(file->reading);
#ifdef _POSIX_C_SOURCE
  int res = getc_unlocked(file->file);
#else
  int res = getc(file->file);
#endif
  if (res != EOF) {
    file->bytes++;
  }
  return res;
}

static inline int kissat_putc(file *file, int ch) {
  assert(file);
  assert(file->file);
  assert(!file->reading);
#ifdef _POSIX_C_SOURCE
  int res = putc_unlocked(ch, file->file);
#else
  int res = putc(ch, file->file);
#endif
  if (res != EOF) {
    file->bytes++;
  }
  return ch;
}

#endif
