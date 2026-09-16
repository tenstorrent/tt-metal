# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Slice captured sidecars according to each router's manifest layout."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .inputs import DecodeInput
from .structs import SUPPORTED_SCHEMAS, decode_payload


def _endpoint_key(endpoint: dict[str, Any]) -> tuple[int, int, int]:
    return (endpoint["mesh_id"], endpoint["chip_id"], endpoint["eth_chan"])


class RegionDecoder:
    """Decode regions while caching each raw sidecar once."""

    def __init__(self, inputs: tuple[DecodeInput, ...], *, expert_raw: bool = False):
        self.inputs = inputs
        self.expert_raw = expert_raw
        self._raw = {
            index: item.raw.path.read_bytes()
            for index, item in enumerate(inputs)
            if item.raw is not None
        }

    def _sample(self, input_index: int, endpoint: dict[str, Any]) -> dict[str, Any]:
        item = self.inputs[input_index]
        assert item.snapshot is not None
        wanted = _endpoint_key(endpoint)
        for sample in item.snapshot["samples"][0]["routers"]:
            if _endpoint_key(sample["id"]) == wanted:
                return sample
        raise ValueError(f"snapshot {input_index} has no router {wanted}")

    def sample_for(self, input_index: int, endpoint: dict[str, Any]) -> dict[str, Any]:
        return self._sample(input_index, endpoint)

    def raw_slice(self, input_index: int, raw_ref: dict[str, Any]) -> bytes | None:
        payload = self._raw.get(input_index)
        if payload is None:
            return None
        offset = raw_ref["offset"]
        return payload[offset : offset + raw_ref["size"]]

    @staticmethod
    def _base(region: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: deepcopy(region[key])
            for key in (
                "id",
                "parent",
                "backing",
                "schema",
                "allocated",
                "enabled",
                "writer",
                "address",
                "size",
                "count",
                "stride",
                "stream_id",
            )
            if key in region
        }
        result.update({"status": "ok", "error": None, "raw_ref": None, "value": None})
        return result

    @staticmethod
    def _covering_blob(sample: dict[str, Any], address: int, size: int):
        end = address + size
        for name, blob in sample.get("blobs", {}).items():
            blob_address = blob.get("address")
            blob_size = blob.get("size")
            if (
                isinstance(blob_address, int)
                and isinstance(blob_size, int)
                and blob_address <= address
                and end <= blob_address + blob_size
            ):
                return name, blob
        return None, None

    def decode_router(self, router: dict[str, Any]) -> list[dict[str, Any]]:
        """Decode all regions for one merged router skeleton."""

        input_index = router["capture"]["snapshot_index"]
        if input_index is None or router["layout_id"] is None:
            return []
        item = self.inputs[input_index]
        manifest = item.manifest.manifest
        sample = self._sample(input_index, router["id"])
        layout = manifest.layouts[router["layout_id"]]
        return [self._decode_region(region, router, sample, input_index) for region in layout["regions"]]

    def _decode_region(
        self,
        region: dict[str, Any],
        router: dict[str, Any],
        sample: dict[str, Any],
        input_index: int,
    ) -> dict[str, Any]:
        result = self._base(region)
        backing = region["backing"]
        schema = region.get("schema")
        if backing == "group":
            return result
        if not region.get("allocated", False) or region.get("size") == 0:
            result["status"] = "unallocated"
            return result

        if backing == "stream_reg":
            stream_id = str(region.get("stream_id"))
            pre = sample.get("streams", {}).get("pre", {}).get(stream_id)
            post = sample.get("streams", {}).get("post", {}).get(stream_id)
            if pre is None or post is None:
                result.update(status="not_captured", error=f"stream {stream_id} was not captured")
            else:
                pre_value = pre.get("buf_space_available")
                post_value = post.get("buf_space_available")
                torn = pre_value != post_value
                result["value"] = {"pre": pre_value, "post": post_value, "torn": torn}
                if torn:
                    result["status"] = "torn"
            return result

        if schema == "heartbeat_word":
            values = [
                {"t": value.get("t"), "raw": value.get("heartbeat")}
                for value in sample.get("liveness", [])
            ]
            result["value"] = {"samples": values}
            if not values:
                result.update(status="not_captured", error="heartbeat samples are absent")
            return result

        address = region.get("address")
        size = region.get("size")
        if not isinstance(address, int) or not isinstance(size, int):
            result.update(status="unsupported", error="L1 region has no integer address/size")
            return result
        blob_name, blob = self._covering_blob(sample, address, size)
        if blob is None:
            result.update(status="not_captured", error="no captured blob covers this region")
            return result
        if blob.get("status") != "ok":
            result.update(status=blob.get("status", "unreadable"), error=blob.get("error"))
            return result
        if input_index not in self._raw or blob.get("offset") is None:
            result.update(status="not_captured", error="raw sidecar is unavailable")
            return result

        raw_offset = int(blob["offset"]) + address - int(blob["address"])
        payload = self._raw[input_index][raw_offset : raw_offset + size]
        result["raw_ref"] = {
            "file": item_name(self.inputs[input_index]),
            "offset": raw_offset,
            "size": size,
            "blob": blob_name,
        }
        if len(payload) != size:
            result.update(status="unreadable", error=f"raw sidecar ended after {len(payload)} of {size} bytes")
            return result
        if self.expert_raw and size <= 64:
            result["raw_hex"] = payload.hex()
        if router["capture"]["status"] == "reset":
            result["status"] = "reset"
            return result
        if schema is None:
            return result
        if schema not in SUPPORTED_SCHEMAS:
            result.update(status="unsupported", error=f"unknown region schema {schema!r}")
            return result
        try:
            result["value"] = decode_payload(
                schema,
                payload,
                enums=self.inputs[input_index].manifest.manifest.enums,
                region=region,
                mesh_count=len(self.inputs[input_index].manifest.manifest.data["meshes"]),
            )
        except (KeyError, ValueError) as error:
            result.update(status="unsupported", error=str(error), value=None)
        return result


def item_name(item: DecodeInput) -> str:
    return item.snapshot["raw"]["file"] if item.snapshot is not None else ""
