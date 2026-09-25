from aiohttp import web
import signal
import json
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


async def handle_post(request):
    name = request.match_info.get("name", "AAAAnonymous")
    data = await request.json()
    eprint(data)
    request.app[db_key].append(data)
    eprint(request.app[db_key])

    text = "Hello, " + name + "\n" + str(data) + "\n"
    return web.Response(text=text)


async def handle_done(request):
    with open("dump.json", "w") as f:
        f.write(json.dumps(request.app[db_key], indent=4))
    request.app[db_key].clear()

    signal.alarm(1)
    return web.Response(text="Success\n")


async def handle(request):
    name = request.match_info.get("name", "Anonymous")
    text = "Hello, " + name + "\n"
    return web.Response(text=text)


db_key = web.AppKey("db_key", [])
app = web.Application()
app.add_routes(
    [
        web.get("/", handle),
        web.get("/{name}", handle),
        web.post("/", handle_post),
        web.post("/done", handle_done),
    ]
)
app[db_key] = []

if __name__ == "__main__":
    web.run_app(app)
