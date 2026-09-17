#include "file.h"
#include "utilities.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

bool kissat_file_exists(const char *path) {
  if (!path) {
    return false;
  }
  struct stat buf;
  if (stat(path, &buf)) {
    return false;
  }
  return true;
}

bool kissat_file_readable(const char *path) {
  if (!path) {
    return false;
  }
  struct stat buf;
  if (stat(path, &buf)) {
    return false;
  }
  if (access(path, R_OK)) {
    return false;
  }
  return true;
}

bool kissat_file_writable(const char *path) {
  int res;
  if (!path) {
    res = 1;
  } else if (!strcmp(path, "/dev/null")) {
    res = 0;
  } else {
    if (!*path) {
      res = 2;
    } else {
      struct stat buf;
      const char *p = strrchr(path, '/');
      if (!p) {
        if (stat(path, &buf)) {
          if (errno == ENOENT) {
            res = 0;
          } else {
            res = -2;
          }
        } else if (S_ISDIR(buf.st_mode)) {
          res = 3;
        } else if (access(path, W_OK)) {
          res = 4;
        } else {
          res = 0;
        }
      } else if (!p[1]) {
        res = 5;
      } else {
        const size_t len = p - path;
        char *dirname = malloc(len + 1);
        if (dirname) {
          strncpy(dirname, path, len);
          dirname[len] = 0;
          if (stat(dirname, &buf)) {
            res = 6;
          } else if (!S_ISDIR(buf.st_mode)) {
            res = 7;
          } else if (access(dirname, W_OK)) {
            res = 8;
          } else if (stat(path, &buf)) {
            if (errno == ENOENT) {
              res = 0;
            } else {
              res = -3;
            }
          } else if (access(path, W_OK)) {
            res = 9;
          } else {
            res = 0;
          }
          free(dirname);
        } else {
          res = 10;
        }
      }
    }
  }
  return !res;
}

size_t kissat_file_size(const char *path) {
  struct stat buf;
  if (stat(path, &buf)) {
    return 0;
  }
  return (size_t) buf.st_size;
}

bool kissat_find_executable(const char *name) {
  const size_t name_len = strlen(name);
  const char *environment = getenv("PATH");
  if (!environment) {
    return false;
  }
  const size_t dirs_len = strlen(environment);
  char *dirs = malloc(dirs_len + 1);
  if (!dirs) {
    return false;
  }
  strcpy(dirs, environment);
  bool res = false;
  const char *end = dirs + dirs_len + 1;
  for (char *dir = dirs, *q; !res && dir != end; dir = q) {
    for (q = dir; *q && *q != ':'; q++) {
      assert(q + 1 < end);
    }
    *q++ = 0;
    const size_t path_len = (q - dir) + name_len;
    char *path = malloc(path_len + 1);
    if (!path) {
      free(dirs);
      return false;
    }
    sprintf(path, "%s/%s", dir, name);
    assert(strlen(path) == path_len);
    res = kissat_file_readable(path);
    free(path);
  }
  free(dirs);
  return res;
}

static int bz2sig[] = { 0x42, 0x5A, 0x68, EOF };
static int gzsig[] = { 0x1F, 0x8B, EOF };
static int lzmasig[] = { 0x5D, 0x00, 0x00, 0x80, 0x00, EOF };
static int sig7z[] = { 0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C, EOF };
static int xzsig[] = { 0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00, EOF };

static bool match_signature(const char *path, const int *sig) {
  assert(path);
  FILE *tmp = fopen(path, "r");
  if (!tmp) {
    return false;
  }
  bool res = true;
  for (const int *p = sig; res && (*p != EOF); p++) {
    res = (getc(tmp) == *p);
  }
  fclose(tmp);
  return res;
}

#ifdef _POSIX_C_SOURCE

static FILE *open_pipe(const char *fmt, const char *path, const char *mode) {
  size_t name_len = 0;
  while (fmt[name_len] && fmt[name_len] != ' ') {
    name_len++;
  }
  char *name = malloc(name_len + 1);
  if (!name) {
    return 0;
  }
  strncpy(name, fmt, name_len);
  name[name_len] = 0;
  bool found = kissat_find_executable(name);
  free(name);
  if (!found) {
    return 0;
  }
  char *cmd = malloc(strlen(fmt) + strlen(path));
  if (!cmd) {
    return 0;
  }
  sprintf(cmd, fmt, path);
  FILE *res = popen(cmd, mode);
  free(cmd);
  return res;
}

static FILE *read_pipe(const char *fmt, const int *sig, const char *path) {
  if (!kissat_file_readable(path)) {
    return 0;
  }
  if (sig && !match_signature(path, sig)) {
    return 0;
  }
  return open_pipe(fmt, path, "r");
}

static FILE *write_pipe(const char *fmt, const char *path) {
  return open_pipe(fmt, path, "w");
}

#endif

void kissat_read_already_open_file(file *file, FILE *f, const char *path) {
  file->file = f;
  file->close = false;
  file->reading = true;
  file->compressed = false;
  file->path = path;
  file->bytes = 0;
}

void kissat_write_already_open_file(file *file, FILE *f, const char *path) {
  file->file = f;
  file->close = false;
  file->reading = false;
  file->compressed = false;
  file->path = path;
  file->bytes = 0;
}

#ifndef _POSIX_C_SOURCE

bool kissat_looks_like_a_compressed_file(const char *path) {
#define RETURN_TRUE_IF_COMPRESSED(SUFFIX,SIGNATURE) \
  if (kissat_has_suffix (path, SUFFIX) && \
      match_signature (path, SIGNATURE)) \
    return true

  RETURN_TRUE_IF_COMPRESSED(".bz2", bz2sig);
  RETURN_TRUE_IF_COMPRESSED(".gz", gzsig);
  RETURN_TRUE_IF_COMPRESSED(".lzma", lzmasig);
  RETURN_TRUE_IF_COMPRESSED(".7z", sig7z);
  RETURN_TRUE_IF_COMPRESSED(".xz", xzsig);

  return false;
}

#endif

bool kissat_open_to_read_file(file *file, const char *path) {
#ifdef _POSIX_C_SOURCE
#define READ_PIPE(SUFFIX, CMD, SIG) \
do { \
  if (kissat_has_suffix (path, SUFFIX)) \
    { \
      file->file = read_pipe (CMD, SIG, path); \
      if (!file->file) \
	break; \
      file->close = true; \
      file->reading = true; \
      file->compressed = true; \
      file->path = path; \
      file->bytes = 0; \
      return true; \
    } \
} while (0)
  READ_PIPE(".bz2", "bzip2 -c -d %s", bz2sig);
  READ_PIPE(".gz", "gzip -c -d %s", gzsig);
  READ_PIPE(".lzma", "lzma -c -d %s", lzmasig);
  READ_PIPE(".7z", "7z x -so %s 2>/dev/null", sig7z);
  READ_PIPE(".xz", "xz -c -d %s", xzsig);
#endif
  file->file = fopen(path, "r");
  if (!file->file) {
    return false;
  }
  file->close = true;
  file->reading = true;
  file->compressed = false;
  file->path = path;
  file->bytes = 0;

  return true;
}

bool kissat_open_to_write_file(file *file, const char *path) {
#ifdef _POSIX_C_SOURCE
#define WRITE_PIPE(SUFFIX, CMD) \
do { \
  if (kissat_has_suffix (path, SUFFIX)) \
    { \
      if (SUFFIX[1] == '7' && kissat_file_readable (path) && unlink (path)) \
	return false; \
      file->file = write_pipe (CMD, path); \
      if (!file->file) \
	return false; \
      file->close = true; \
      file->reading = false; \
      file->compressed = true; \
      file->path = path; \
      file->bytes = 0; \
      return true; \
    } \
} while (0)
  WRITE_PIPE(".bz2", "bzip2 -c > %s");
  WRITE_PIPE(".gz", "gzip -c > %s");
  WRITE_PIPE(".lzma", "lzma -c > %s");
  WRITE_PIPE(".7z", "7z a -si %s 2>/dev/null");
  WRITE_PIPE(".xz", "xz -c > %s");
#endif
  file->file = fopen(path, "w");
  if (!file->file) {
    return false;
  }
  file->close = true;
  file->reading = false;
  file->compressed = false;
  file->path = path;
  file->bytes = 0;
  return true;
}

void kissat_close_file(file *file) {
  assert(file);
  assert(file->file);
#ifdef _POSIX_C_SOURCE
  if (file->close && file->compressed) {
    pclose(file->file);
  }
#else
  assert(!file->compressed);
#endif
  if (file->close && !file->compressed) {
    fclose(file->file);
  }
  file->file = 0;
}
