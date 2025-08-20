import threading
import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import asyncio
import websockets


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[{self.client_address[0]}] {format % args}")


# WebSocket notification logic
WS_PORT = 8081
EMBED_FILE = "embed.tracy"
clients = set()


async def notify_clients():
    print("Embed file changed, notifying clients...")
    if clients:
        print(f"Notifying {clients} clients")
        await asyncio.wait([client.send("reload") for client in clients])


async def watch_embed_file():
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(EMBED_FILE)
            if last_mtime is None:
                last_mtime = mtime
            elif mtime != last_mtime:
                last_mtime = mtime
                await notify_clients()
        except FileNotFoundError:
            pass
        await asyncio.sleep(1)


async def ws_handler(websocket):
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.remove(websocket)


async def websocket_main():
    ws_server = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    print(f"WebSocket server running on ws://0.0.0.0:{WS_PORT}")
    await watch_embed_file()


def start_websocket_server():
    asyncio.run(websocket_main())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve WASM files with COOP/COEP headers.")
    parser.add_argument("--dir", type=str, default=os.getcwd(), help="Directory to serve (default: current directory)")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    args = parser.parse_args()

    os.chdir(args.dir)
    print(f"Serving WASM from {os.getcwd()} on http://0.0.0.0:{args.port} ...")
    # Start WebSocket server in a separate thread
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    # Start HTTP server
    HTTPServer(("0.0.0.0", args.port), CORSRequestHandler).serve_forever()
