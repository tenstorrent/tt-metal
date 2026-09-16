const DECODED_KIND = "fabric_debug_decoded";
const DECODED_VERSION = 1;

export class LoadError extends Error {
  constructor(message) {
    super(message);
    this.name = "LoadError";
  }
}

function requireObject(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new LoadError(`${label} must be an object`);
  }
}

export function validateDecoded(decoded) {
  requireObject(decoded, "Decoded state");

  if (decoded.kind !== DECODED_KIND) {
    throw new LoadError(
      `Expected kind "${DECODED_KIND}", got ${JSON.stringify(decoded.kind)}`,
    );
  }
  if (decoded.decoded_version !== DECODED_VERSION) {
    throw new LoadError(
      `Unsupported decoded_version ${JSON.stringify(decoded.decoded_version)}; expected ${DECODED_VERSION}`,
    );
  }
  requireObject(decoded.run, "run");
  requireObject(decoded.fabric_context, "fabric_context");
  requireObject(decoded.coverage, "coverage");
  requireObject(decoded.topology, "topology");
  if (!Array.isArray(decoded.topology.meshes) || !Array.isArray(decoded.topology.links)) {
    throw new LoadError("topology.meshes and topology.links must be arrays");
  }
  if (!Array.isArray(decoded.routers)) {
    throw new LoadError("routers must be an array");
  }
  return decoded;
}

export function parseDecoded(text) {
  let decoded;
  try {
    decoded = JSON.parse(text);
  } catch (error) {
    throw new LoadError(`Invalid JSON: ${error.message}`);
  }
  return validateDecoded(decoded);
}

export async function loadDecodedFile(file) {
  if (!(file instanceof File)) {
    throw new LoadError("Drop or choose one JSON file");
  }
  let text;
  try {
    text = await file.text();
  } catch (error) {
    throw new LoadError(`Could not read ${file.name}: ${error.message}`);
  }
  return {
    decoded: parseDecoded(text),
    sourceName: file.name,
  };
}

export async function fetchDecoded(url) {
  let response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch (error) {
    throw new LoadError(`Could not fetch ${url}: ${error.message}`);
  }
  if (!response.ok) {
    throw new LoadError(`Could not fetch ${url}: HTTP ${response.status}`);
  }
  return {
    decoded: parseDecoded(await response.text()),
    sourceName: url.split("/").at(-1) || url,
  };
}
