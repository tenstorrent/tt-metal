#!/usr/bin/env python3
"""Loopback-only transparent transport to the owned Docker bridge server."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys


async def main():
    target=subprocess.check_output(['docker','inspect','qwen38-functional-build','--format',
        '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],text=True).strip()
    async def relay(reader,writer):
        try:
            while chunk:=await reader.read(65536):
                writer.write(chunk)
                await writer.drain()
        finally:
            writer.close()
    async def connection(reader,writer):
        try:
            upstream,output=await asyncio.open_connection(target,8000)
            await asyncio.gather(relay(reader,output),relay(upstream,writer))
        except (ConnectionError,OSError):
            writer.close()
    server=await asyncio.start_server(connection,'127.0.0.1',8000)
    Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        listen='127.0.0.1:8000',upstream=target+':8000'))+'\n')
    async with server:
        await server.serve_forever()


asyncio.run(main())
