#include "file.h"
#include "message.h"
#include "stack.h"

#include <assert.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>

void write_buffer (struct buffer *buffer, struct file *file) {
  assert (file);
  if (file->lock)
    acquire_message_lock ();
  size_t size = SIZE (*buffer);
  fwrite (buffer->begin, size, 1, file->file);
  if (file->lock)
    release_message_lock ();
  CLEAR (*buffer);
  atomic_fetch_add (&file->lines, 1);
}

void close_proof (struct file *proof) {
  if (!proof->file)
    return;
  if (proof->close) {
    fclose (proof->file);
    if (verbosity >= 0)
      printf ("c\nc closed '%s' after writing %" PRIu64 " proof lines\n",
              proof->path, proof->lines);
  } else if (verbosity >= 0)
    printf ("c\nc finished writing %" PRIu64 " proof lines to '%s'\n",
            proof->lines, proof->path);

  if (verbosity >= 0)
    fflush (stdout);
}
