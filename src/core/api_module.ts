// src/core/api_module.ts
export function processIncomingContext(req: Request, res: Response) {
  // FIX: Sanitization checks to enforce standard input formats
  const sanitizedInput = sanitizeReqBody(req.body);
  if (!sanitizedInput.isValid) {
    return res.status(400).json({ error: "Context validation failed" });
  }
  return executeContextHandler(sanitizedInput);
}