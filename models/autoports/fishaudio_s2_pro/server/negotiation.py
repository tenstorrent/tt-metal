"""Body parsing (JSON / application/msgpack / multipart) and response-format selection, fish-speech style."""
from __future__ import annotations

import json
from typing import Any, Dict, Type

from fastapi import HTTPException, Request
from pydantic import BaseModel, ValidationError


async def parse_body(request: Request, model: Type[BaseModel]) -> BaseModel:
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    raw = await request.body()
    if ctype == "application/msgpack":
        try:
            import ormsgpack
        except ImportError as e:
            raise HTTPException(415, f"msgpack not available on this server: {e}")
        data: Dict[str, Any] = ormsgpack.unpackb(raw)
    elif ctype in ("application/json", "", "text/plain"):
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"invalid JSON: {e}")
    elif ctype == "multipart/form-data":
        form = await request.form()
        data = {}
        for k, v in form.multi_items():
            if hasattr(v, "read"):
                data[k] = await v.read()
            else:
                data[k] = v
    else:
        raise HTTPException(
            415,
            f"unsupported Content-Type {ctype!r}; use application/json, application/msgpack or multipart/form-data",
            headers={"Accept": "application/json, application/msgpack, multipart/form-data"},
        )
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False))


def wants_json(request: Request) -> bool:
    q = (request.query_params.get("format") or "").strip().lower()
    if q in ("json", "msgpack"):
        return q == "json"
    accept = (request.headers.get("accept") or "").lower()
    if "application/msgpack" in accept and "application/json" not in accept:
        return False
    return True


def pack_response(request: Request, obj: BaseModel):
    from fastapi.responses import JSONResponse, Response

    if wants_json(request):
        return JSONResponse(obj.model_dump())
    try:
        import ormsgpack

        return Response(ormsgpack.packb(obj.model_dump()), media_type="application/msgpack")
    except ImportError:
        return JSONResponse(obj.model_dump())
