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
  const decoded = parseDecoded(text);
  return {
    decoded,
    sourceName: file.name,
    rawFiles: new Map(),
    rawBaseUrl: null,
    rawManifest: indexRawManifest(decoded),
  };
}

function rawKeyCandidates(file) {
  const keys = [file.name];
  const relative = file.webkitRelativePath || "";
  if (relative) {
    // Folder picks report "folder/sub/dir/file.bin"; sidecars may live in a
    // subdirectory, so index both the full relative path and the basename.
    keys.push(relative);
    const parts = relative.split("/");
    if (parts.length > 1) {
      keys.push(parts.slice(1).join("/"));
    }
  }
  return [...new Set(keys)];
}

function indexRawFiles(files) {
  const rawFiles = new Map();
  for (const file of files) {
    if (!file.name.endsWith(".bin")) {
      continue;
    }
    for (const key of rawKeyCandidates(file)) {
      if (!rawFiles.has(key)) {
        rawFiles.set(key, file);
      }
    }
    const base = file.name;
    if (!rawFiles.has(base)) {
      rawFiles.set(base, file);
    }
  }
  return rawFiles;
}

/**
 * Load a decode output folder: one decoded `.json` plus its `.bin` sidecar.
 * Names don't matter. Accepts a FileList from an `<input webkitdirectory>`
 * picker or a folder drag-drop. Raw sidecars stay as lazy `File` handles;
 * only the decoded JSON is parsed.
 */
export async function loadDecodedFromFileList(fileList) {
  const files = [...(fileList || [])].filter((file) => file instanceof File);
  if (!files.length) {
    throw new LoadError("That folder has no files");
  }
  const jsonFiles = files.filter((file) => file.name.endsWith(".json"));
  if (jsonFiles.length === 0) {
    throw new LoadError("That folder has no JSON file");
  }
  let decoded = null;
  let decodedFile = null;
  if (jsonFiles.length === 1) {
    // The common case: folder holds one JSON + its BIN, whatever they're called.
    decodedFile = jsonFiles[0];
    decoded = parseDecoded(await decodedFile.text());
  } else {
    let lastError = null;
    const seen = new Set();
    const preferred = [
      ...jsonFiles.filter((file) => file.name.includes("decoded")),
      ...jsonFiles,
    ];
    for (const file of preferred) {
      if (seen.has(file)) {
        continue;
      }
      seen.add(file);
      try {
        decoded = parseDecoded(await file.text());
        decodedFile = file;
        break;
      } catch (error) {
        lastError = error;
      }
    }
    if (!decoded || !decodedFile) {
      throw new LoadError(
        `No fabric_debug_decoded JSON in that folder${lastError ? `: ${lastError.message}` : ""}`,
      );
    }
  }
  const relative = decodedFile.webkitRelativePath || "";
  const folderName = relative ? relative.split("/")[0] : "folder";
  return {
    decoded,
    sourceName: `${folderName}/${decodedFile.name}`,
    rawFiles: indexRawFiles(files),
    rawBaseUrl: null,
    rawManifest: indexRawManifest(decoded),
  };
}

/**
 * Index decoded raw_files[] by sidecar name. Duplicates are kept: two ranks
 * may share a stem, and collapsing them would hide a gather-time collision
 * from the verifier in chip.js.
 */
export function indexRawManifest(decoded) {
  const table = new Map();
  for (const entry of decoded?.raw_files || []) {
    if (!entry || typeof entry.file !== "string") {
      continue;
    }
    if (!table.has(entry.file)) {
      table.set(entry.file, []);
    }
    table.get(entry.file).push({ size: entry.size, sha256: entry.sha256 });
  }
  return table;
}

export function resolveRawFile(rawFiles, refFile) {
  if (!rawFiles || rawFiles.size === 0) {
    return null;
  }
  if (refFile) {
    const direct =
      rawFiles.get(refFile) ||
      rawFiles.get(String(refFile).split("/").at(-1)) ||
      null;
    if (direct) {
      return direct;
    }
  }
  // Folder holds one JSON + its BIN with arbitrary names: a lone sidecar
  // satisfies any ref.
  const bins = [...rawFiles.values()];
  const unique = [...new Set(bins)];
  return unique.length === 1 ? unique[0] : null;
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
  const rawBaseUrl = url.includes("/") ? url.slice(0, url.lastIndexOf("/") + 1) : "";
  const decoded = parseDecoded(await response.text());
  return {
    decoded,
    sourceName: url.split("/").at(-1) || url,
    rawFiles: new Map(),
    rawBaseUrl,
    rawManifest: indexRawManifest(decoded),
  };
}
