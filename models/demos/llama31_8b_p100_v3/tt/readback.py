# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class PipelinedDecode:
    def __init__(self, generator, is_tokens=True):
        self.generator = generator
        self.is_tokens = is_tokens
        self._pending = None

    def step(self, tokens, start_pos, **kwargs):
        kwargs.pop("read_from_device", None)
        tt_out = self.generator.decode_forward(tokens, start_pos, read_from_device=False, **kwargs)
        host, events = self.generator.read_decode_output(tt_out, async_read=True)
        previous = self.flush()
        self._pending = (host, events)
        return previous

    def flush(self):
        if self._pending is None:
            return None
        host, events = self._pending
        self._pending = None
        for event in events:
            ttnn.event_synchronize(event)
        return self.generator.process_decode_output_host(host, is_tokens=self.is_tokens)
